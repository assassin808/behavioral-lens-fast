#!/bin/bash
# Tests 2+4, one-time LOGIN-node prep (needs internet). Run from a checkout of branch yang/t2-t4 with the env active:
#   source $BLF_HOME/env/bin/activate; BLF_DATA=$BLF_HOME/data bash scripts/portable_t24_prep.sh
# Installs anthropics/jacobian-lens (jlens, Apache-2.0, pinned commit) and downloads into $BLF_DATA:
#   ext/jlens, ext/jpp  : Neuronpedia's pre-fitted J-lens / J++ lens for Qwen3.5-0.8B (HF neuronpedia/jacobian-lens)
#   ext/sae_l17         : public SAE decoderesearch/qwen-3.5-saes qwen-3.5-0.8b/btk-mat-layer-17-k-100
#   wikitext/train0.parquet : Salesforce/wikitext wikitext-103-raw-v1 train shard 0 (normal-text corpus)
set -euo pipefail
: "${BLF_DATA:?set BLF_DATA}"
python -c "import jlens" 2>/dev/null || pip install -q "git+https://github.com/anthropics/jacobian-lens.git@581d398"
python -c "import pyarrow" 2>/dev/null || pip install -q pyarrow
mkdir -p "$BLF_DATA"/ext/{jlens,jpp,sae_l17} "$BLF_DATA"/wikitext
N=https://huggingface.co/neuronpedia/jacobian-lens/resolve/main/qwen3.5-0.8b
S=https://huggingface.co/decoderesearch/qwen-3.5-saes/resolve/main/qwen-3.5-0.8b/btk-mat-layer-17-k-100
get() { [ -s "$2" ] || curl -sfL -o "$2" "$1"; }
get $N/jlens/Salesforce-wikitext/Qwen3.5-0.8B_jacobian_lens.pt "$BLF_DATA/ext/jlens/Qwen3.5-0.8B_jacobian_lens.pt"
get $N/jpp/Salesforce-wikitext/Qwen3.5-0.8B_jpp_lens.pt "$BLF_DATA/ext/jpp/Qwen3.5-0.8B_jpp_lens.pt"
for f in cfg.json runner_cfg.json sae_weights.safetensors; do get $S/$f "$BLF_DATA/ext/sae_l17/$f"; done
get https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-103-raw-v1/train-00000-of-00002.parquet "$BLF_DATA/wikitext/train0.parquet"
ls -la "$BLF_DATA"/ext/* "$BLF_DATA"/wikitext
python -c "import jlens, pyarrow; print('jlens + pyarrow OK')"
