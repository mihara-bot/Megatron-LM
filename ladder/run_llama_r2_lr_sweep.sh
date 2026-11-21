#!/bin/bash
# Submit the Llama3 R2 Muon sweep from a shared checkout with the uv environment active.
# Preview all submissions: DRY_RUN=1 bash ladder/run_llama_r2_lr_sweep.sh
# Screen all five LRs through TPP 20: UPTO=20 bash ladder/run_llama_r2_lr_sweep.sh
# Submit a single LR on idle nodes: QOS=polaris_idle LRS=1.3e-3 bash ladder/run_llama_r2_lr_sweep.sh

export CONFIG=llama3_r2 OPT=muon
# At fixed TPP, C scales roughly as N_eff^2. The config's C^-0.125 LR
# prior shifts the R1 grid by ~0.90. Keep the three R2 config points
# and bracket them with the shifted R1 endpoints for this Muon sweep.
export LRS="${LRS:-4.5e-4 6e-4 9e-4 1.3e-3 1.8e-3}"
exec bash "$(dirname "${BASH_SOURCE[0]}")/sweep.sh"
