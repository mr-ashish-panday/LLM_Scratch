#!/bin/bash
# =============================================
# ESH Training — Push Everything to GitHub + HF
# Usage: bash push_all.sh GH_TOKEN HF_TOKEN
# =============================================

GH_TOKEN=$1
HF_TOKEN=$2

if [ -z "$GH_TOKEN" ]; then
  echo "ERROR: Provide GitHub token as first argument"
  echo "Usage: bash push_all.sh GITHUB_TOKEN HF_TOKEN"
  exit 1
fi

echo "=== Pushing to GitHub ==="
git remote set-url origin https://mr-ashish-panday:${GH_TOKEN}@github.com/mr-ashish-panday/LLM_Scratch.git
git add -f width_only_train.log 2>/dev/null || true
git add -f baseline_train.log 2>/dev/null || true
git add results/ 2>/dev/null || true
git commit -m "Add H200 training logs, metrics and results" 2>/dev/null || echo "Nothing new to commit"
git push origin main
echo "GitHub push done!"

if [ -z "$HF_TOKEN" ]; then
  echo "Skipping Hugging Face (no token provided)"
  exit 0
fi

echo ""
echo "=== Pushing checkpoint to Hugging Face ==="
pip install huggingface_hub -q
python3 - <<EOF
from huggingface_hub import HfApi
import os
api = HfApi()
api.create_repo("mr-ashish-panday/esh-checkpoints", token="${HF_TOKEN}", exist_ok=True, repo_type="model")
# Upload width_only checkpoint if it exists
import glob
ckpts = glob.glob("results/width_only/*.pt") + glob.glob("results/baseline/*.pt")
for ckpt in ckpts:
    print(f"Uploading {ckpt}...")
    api.upload_file(path_or_fileobj=ckpt, path_in_repo=ckpt, repo_id="mr-ashish-panday/esh-checkpoints", token="${HF_TOKEN}", repo_type="model")
    print(f"Done: {ckpt}")
print("HuggingFace upload complete!")
EOF
