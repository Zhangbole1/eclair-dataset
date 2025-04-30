#!/usr/bin/env bash
set -e

echo ">>> Checking CUDA in container …"
python3 - <<EOF
import torch
if not torch.cuda.is_available():
    print("ERROR: CUDA not available!")
print("SUCCESS: CUDA is available, GPU count =", torch.cuda.device_count())
EOF

# 如果你还要跑其它命令，追加在这里：
exec "$@"
