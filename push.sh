#!/bin/bash
# Usage: bash push.sh YOUR_GITHUB_TOKEN
TOKEN=$1
git remote set-url origin https://mr-ashish-panday:${TOKEN}@github.com/mr-ashish-panday/LLM_Scratch.git
git add -f width_only_train.log
git add results/
git commit -m "Add H200 width_only training logs and results" 2>/dev/null || echo "Nothing new to commit"
git push origin main
echo "Done!"
