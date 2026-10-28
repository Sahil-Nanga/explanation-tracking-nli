#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

J2=$(jobs/submit.sh roihu_m2 dev,main,noise,revise --parsable)
J1=$(jobs/submit.sh roihu_m1 dev,main,noise,revise --parsable)
JP=$(jobs/submit.sh roihu_m1 probe --parsable --dependency="afterok:$J2:$J1")
echo "submitted: m2=$J2 m1=$J1 probe=$JP"
echo "watch with: squeue --me"
