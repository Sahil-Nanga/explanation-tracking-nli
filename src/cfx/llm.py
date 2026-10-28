import hashlib
import http.client
import json
import os
import random
import socket
import time
from pathlib import Path
from urllib.parse import urlparse

from .config import CACHE


class _UnixConnection(http.client.HTTPConnection):
    def __init__(self, path, timeout):
        super().__init__("localhost", timeout=timeout)
        self._path = path

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self._path)


class Endpoint:
    def __init__(self, uds=None, base_url=None, api_key=None, timeout=600):
        self.uds = uds if uds is not None else os.environ.get("CFX_UDS")
        url = urlparse(base_url or os.environ.get("CFX_BASE_URL", "http://localhost:8000/v1"))
        self.scheme, self.host, self.port = url.scheme, url.hostname, url.port
        self.prefix = url.path.rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("CFX_API_KEY", "")
        self.timeout = timeout

    def describe(self):
        return f"unix:{self.uds}" if self.uds else f"{self.scheme}://{self.host}{f':{self.port}' if self.port else ''}{self.prefix}"

    def _conn(self):
        if self.uds:
            return _UnixConnection(self.uds, self.timeout)
        cls = http.client.HTTPSConnection if self.scheme == "https" else http.client.HTTPConnection
        return cls(self.host, self.port, timeout=self.timeout)

    def request(self, method, path, payload=None):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        conn = self._conn()
        try:
            body = json.dumps(payload) if payload is not None else None
            conn.request(method, self.prefix + path, body=body, headers=headers)
            resp = conn.getresponse()
            data = resp.read().decode("utf-8", errors="replace")
            return resp.status, data
        finally:
            conn.close()

    def server_info(self):
        info = {"endpoint": self.describe()}
        for key, path in (("models", "/models"),):
            try:
                status, data = self.request("GET", path)
                if status == 200:
                    ids = [m.get("id") for m in json.loads(data).get("data", [])]
                    info[key] = ids if len(ids) <= 20 else f"{len(ids)} models (hosted API)"
            except (OSError, ValueError, http.client.HTTPException):
                pass
        try:
            saved, self.prefix = self.prefix, ""
            status, data = self.request("GET", "/version")
            if status == 200:
                info["server_version"] = json.loads(data).get("version")
        except (OSError, ValueError, http.client.HTTPException):
            pass
        finally:
            self.prefix = saved
        return info

    def wait_ready(self, timeout=1800, interval=10):
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                status, _ = self.request("GET", "/models")
                if status == 200:
                    return True
            except (OSError, http.client.HTTPException):
                pass
            time.sleep(interval)
        return False

    def chat(self, model, messages, max_tokens, decoding, extra_body=None, max_attempts=8):
        payload = {"model": model, "messages": messages, "max_tokens": max_tokens, "stream": False}
        payload.update(decoding)
        payload.update(extra_body or {})
        last = None
        for attempt in range(max_attempts):
            t0 = time.time()
            try:
                status, data = self.request("POST", "/chat/completions", payload)
            except (OSError, http.client.HTTPException) as e:
                status, data = None, str(e)
            latency = (time.time() - t0) * 1000
            if status == 200:
                try:
                    obj = json.loads(data)
                    choice = obj["choices"][0]
                except (ValueError, KeyError, IndexError, TypeError):
                    obj = None
                if obj is not None:
                    return {"text": choice["message"].get("content") or "",
                            "finish_reason": choice.get("finish_reason"),
                            "usage": obj.get("usage", {}),
                            "latency_ms": round(latency, 1),
                            "server_model": obj.get("model"),
                            "provider": obj.get("provider")}
            last = f"HTTP {status}: {data[:300]}"
            if status is not None and 400 <= status < 500 and status not in (408, 409, 429):
                break
            time.sleep(min(60, 2 ** attempt) + random.random())
        raise RuntimeError(f"request failed after retries: {last}")


class Cache:
    def __init__(self, root=CACHE):
        self.root = Path(root)

    @staticmethod
    def key(model_cfg, messages, max_tokens, decoding, salt=""):
        blob = json.dumps({"model": model_cfg["model"], "revision": model_cfg.get("revision"),
                           "extra_body": model_cfg.get("extra_body", {}), "messages": messages,
                           "max_tokens": max_tokens, "decoding": decoding, "salt": salt},
                          sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()

    def _path(self, model_cfg, key):
        slug = model_cfg["model"].replace("/", "__")
        return self.root / slug / key[:2] / f"{key}.json"

    def get(self, model_cfg, key):
        p = self._path(model_cfg, key)
        if p.exists():
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        return None

    def put(self, model_cfg, key, value):
        p = self._path(model_cfg, key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(f".{os.getpid()}.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False)
        tmp.replace(p)
