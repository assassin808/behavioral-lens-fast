#!/bin/bash
# One-time setup on an Alliance cluster (Nibi / Fir / Rorqual / Narval / Trillium). Run on a LOGIN node
# (needs internet: GitHub, PyPI, Hugging Face). Everything goes under $BLF_HOME (default ~/scratch/blf_run).
set -euo pipefail
BLF_HOME=${BLF_HOME:-$HOME/scratch/blf_run}
mkdir -p "$BLF_HOME" && cd "$BLF_HOME"

module load StdEnv/2023 2>/dev/null || true

# Code: main + collaborator branch + our analysis branches
[ -d repo ] || git clone https://github.com/assassin808/behavioral-lens-fast.git repo
cd repo
git fetch origin '+refs/heads/*:refs/remotes/origin/*'
git fetch https://github.com/xuan7zhang/behavioral-lens.git xuanqi:xuanqi || true
cd ..

# Python env via uv straight from PyPI (Alliance pip + wheelhouse backtracks and builds from source for ages;
# transformers>=5.11 is needed for qwen3_5). uv installs a standalone Python, so no module python is needed.
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; }
export PATH=$HOME/.local/bin:$PATH
unset PIP_CONFIG_FILE PYTHONPATH
[ -d env ] || uv venv env --python 3.11 --python-preference only-managed
source env/bin/activate
UVI="uv pip install --index-url https://pypi.org/simple --index-strategy unsafe-best-match"
$UVI "torch==2.9.1" "transformers>=5.11" safetensors accelerate numpy scipy scikit-learn matplotlib openai huggingface_hub pyarrow
$UVI flash-linear-attention || echo "flash-linear-attention failed: slow torch fallback will be used"
$UVI causal-conv1d --no-build-isolation || echo "causal-conv1d failed: slow fallback will be used"

# Model at Astra's revision, then the p50c edit (apply_edit.py checks sha256)
export HF_HOME=$BLF_HOME/hf_cache
hf download Qwen/Qwen3.5-0.8B --revision 2fc06364715b967f1860aea9cf38778875588b17 --local-dir models/base
mkdir -p models/p50c
# copy configs/tokenizer but not the shard index (p50c is written as a single model.safetensors)
for f in models/base/*.json models/base/*.txt models/base/*.jinja; do
  [ -e "$f" ] || continue; case "$f" in *safetensors.index.json) continue;; esac; cp "$f" models/p50c/
done
BASE_ST=$(ls models/base/*.safetensors | head -1)   # HF names it model.safetensors-00001-of-00001.safetensors
[ "$(ls models/base/*.safetensors | wc -l)" = 1 ] || { echo "expected one base shard"; exit 1; }
python repo/astra_logs/exp8_math/code/apply_edit.py "$BASE_ST" \
    repo/astra_logs/exp8_math/runs/p50c/edit models/p50c/model.safetensors
python -c "from transformers import AutoModelForCausalLM as M; M.from_pretrained('models/p50c'); print('p50c loads')"

python -c "import torch,transformers;print('torch',torch.__version__,'transformers',transformers.__version__)"
echo "Setup done in $BLF_HOME. Next: put VECTOR_API_KEY / VECTOR_BASE_URL / HF_TOKEN in ~/.env, then submit alliance/job.sh."
