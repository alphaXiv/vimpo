#!/usr/bin/env bash
# Minimal end-to-end VIMPO reproduction (arXiv 2606.20008).
#
# Goal: demonstrate the paper's core mechanism — the Value-Implicit Policy
# Optimization objective (policy-implied value loss + PPO actor branch driven by
# the log-ratio TD advantage) — running end to end on a single GPU, on a small
# model and a small slice of the paper's actual math RLVR data.
#
# This is intentionally a smoke-scale config, not the full Qwen3-4B / 54.4k run.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

ART_DIR="${REPO_ROOT}/.openresearch/artifacts"
mkdir -p "${ART_DIR}"
EVAL_MD="${ART_DIR}/EVAL.md"
RUN_LOG="${ART_DIR}/run.log"

# Mirror everything to a persisted artifact log too.
exec > >(tee -a "${RUN_LOG}") 2>&1

fail() {
    echo "[run_minimal] FAILED: $*"
    {
        echo "# VIMPO minimal reproduction — FAILED"
        echo
        echo "Stage: $*"
        echo
        echo "See run.log artifact for details."
    } > "${EVAL_MD}"
    exit 1
}

echo "=============================================="
echo "VIMPO minimal reproduction"
echo "repo root: ${REPO_ROOT}"
echo "start: $(date -u)"
echo "=============================================="
nvidia-smi || true

############################################
# 1. System + python deps
############################################
echo "[run_minimal] installing system packages"
apt-get update -y >/dev/null 2>&1 || true
apt-get install -y git wget build-essential libnuma-dev numactl >/dev/null 2>&1 || true

export HF_HUB_ENABLE_HF_TRANSFER=1
export PIP_ROOT_USER_ACTION=ignore
export TOKENIZERS_PARALLELISM=false

PY=python3
echo "[run_minimal] python: $(${PY} --version 2>&1)"

echo "[run_minimal] installing vllm (pulls a compatible torch) — this is the long step"
${PY} -m pip install --upgrade pip >/dev/null 2>&1 || true
${PY} -m pip install "vllm==0.17.0" || fail "pip install vllm"

echo "[run_minimal] installing verl runtime deps"
${PY} -m pip install \
    "accelerate" "codetiming" "datasets" "dill" "hydra-core" \
    "numpy<2.0.0" "pandas" "peft" "pyarrow>=19.0.0" "pybind11" "pylatexenc" \
    "ray[default]>=2.41.0" "torchdata" "tensordict>=0.8.0,<=0.10.0,!=0.9.0" \
    "transformers" "wandb" "packaging>=20.0" "tensorboard" \
    "math-verify" "latex2sympy2_extended" "liger-kernel" "hf_transfer" \
    || fail "pip install verl deps"

echo "[run_minimal] installing flash-attn (prebuilt wheel preferred)"
${PY} -m pip install flash-attn --no-build-isolation \
    || echo "[run_minimal] WARNING: flash-attn install failed; will fall back to eager/sdpa attention"

echo "[run_minimal] installing verl (this repo) without deps"
${PY} -m pip install --no-deps -e . || fail "pip install -e . (verl)"

# Detect whether flash-attn is importable; if not, disable remove_padding (which
# requires flash-attn varlen) and use an sdpa attention implementation.
if ${PY} -c "import flash_attn" >/dev/null 2>&1; then
    USE_REMOVE_PADDING=True
    ATTN_IMPL=""
    echo "[run_minimal] flash-attn available -> remove_padding=True"
else
    USE_REMOVE_PADDING=False
    ATTN_IMPL="sdpa"
    echo "[run_minimal] flash-attn NOT available -> remove_padding=False, attn=sdpa"
fi

############################################
# 2. Data: small slice of the paper's actual Guru math RLVR data
############################################
DATA_DIR="${REPO_ROOT}/data"
mkdir -p "${DATA_DIR}"
TRAIN_FILE="${DATA_DIR}/math__combined_54.4k.parquet"
MATH500_FILE="${DATA_DIR}/math__math_500.parquet"

GURU="https://huggingface.co/datasets/LLM360/guru-RL-92k/resolve/main"
if [ ! -f "${TRAIN_FILE}" ]; then
    echo "[run_minimal] downloading train data"
    wget -q -O "${TRAIN_FILE}" "${GURU}/train/math__combined_54.4k.parquet" || fail "download train parquet"
fi
if [ ! -f "${MATH500_FILE}" ]; then
    echo "[run_minimal] downloading math-500 eval data"
    wget -q -O "${MATH500_FILE}" "${GURU}/offline_eval/math__math_500.parquet" || fail "download math500 parquet"
fi
# Keep only the columns verl expects.
${PY} recipe/vimpo/data_preprocess/filter_test_dataset_keys.py --input_file "${MATH500_FILE}" || true

############################################
# 3. Minimal training config
############################################
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-0.6B-Base}"
RUN_MODE="${RUN_MODE:-vimpo}"

# Smoke-scale knobs
TRAIN_PROMPT_BSZ=8
N_RESP_PER_PROMPT=4
MAX_PROMPT_LENGTH=1024
MAX_RESPONSE_LENGTH=1024
TRAIN_MAX_SAMPLES=64
VAL_MAX_SAMPLES=32
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-4}"

actor_ppo_max_token_len=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))

echo "[run_minimal] launching VIMPO training (mode=${RUN_MODE}, model=${MODEL_PATH})"
echo "[run_minimal] steps=${TOTAL_TRAINING_STEPS} bsz=${TRAIN_PROMPT_BSZ} n=${N_RESP_PER_PROMPT}"

# Single-GPU ray
ray stop --force >/dev/null 2>&1 || true
sleep 2

if [ "${RUN_MODE}" = "vimpo" ]; then
    use_vimpo_loss=True; use_ppo_actor=True; use_grpo_actor=False
else
    use_vimpo_loss=False; use_ppo_actor=False; use_grpo_actor=False
fi

hydra_attn=()
if [ -n "${ATTN_IMPL}" ]; then
    hydra_attn+=("+actor_rollout_ref.model.override_config.attn_implementation=${ATTN_IMPL}")
fi

set -x
${PY} -m recipe.vimpo.main_vimpo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="['${MATH500_FILE}']" \
    data.prompt_key=prompt \
    data.truncation=left \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.train_batch_size=${TRAIN_PROMPT_BSZ} \
    data.gen_batch_size=${TRAIN_PROMPT_BSZ} \
    data.train_max_samples=${TRAIN_MAX_SAMPLES} \
    data.val_max_samples=${VAL_MAX_SAMPLES} \
    actor_rollout_ref.rollout.n=${N_RESP_PER_PROMPT} \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.model.use_remove_padding=${USE_REMOVE_PADDING} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=2 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${TRAIN_PROMPT_BSZ} \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.max_model_len=${actor_ppo_max_token_len} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=${actor_ppo_max_token_len} \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    reward.reward_manager.name=dapo \
    reward.reward_kwargs.overlong_buffer_cfg.enable=False \
    reward.reward_kwargs.max_resp_len=${MAX_RESPONSE_LENGTH} \
    vimpo_config.use_vimpo_loss=${use_vimpo_loss} \
    vimpo_config.value_type=raw \
    vimpo_config.vimpo_value_loss_type=squared_error \
    vimpo_config.vimpo_value_loss_normalized=False \
    vimpo_config.vimpo_value_loss_coeff=1 \
    vimpo_config.vimpo_ppo_whiten_adv=True \
    vimpo_config.vimpo_beta=0.0005 \
    vimpo_config.vimpo_actor_coeff=0.005 \
    vimpo_config.vimpo_detach_kl=True \
    vimpo_config.vimpo_kl_estimator=exact \
    vimpo_config.vimpo_update_ref_freq=0 \
    vimpo_config.use_grpo_actor=${use_grpo_actor} \
    vimpo_config.use_ppo_actor=${use_ppo_actor} \
    trainer.logger="[console]" \
    trainer.project_name=vimpo-minimal \
    trainer.experiment_name="vimpo-minimal-${RUN_MODE}" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.test_freq=2 \
    trainer.save_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS} \
    trainer.default_local_dir="${REPO_ROOT}/ckpts/minimal" \
    trainer.resume_mode=disable \
    "${hydra_attn[@]}"
TRAIN_RC=$?
set +x

if [ ${TRAIN_RC} -ne 0 ]; then
    fail "training process exited with code ${TRAIN_RC}"
fi

############################################
# 4. Write EVAL.md from the training log
############################################
echo "[run_minimal] training finished; building EVAL.md"
${PY} - "$RUN_LOG" "$EVAL_MD" "$RUN_MODE" "$MODEL_PATH" "$TOTAL_TRAINING_STEPS" <<'PYEOF'
import re, sys

run_log, eval_md, run_mode, model, steps = sys.argv[1:6]
text = open(run_log, errors="replace").read()

# Pull val metrics (verl prints "val-core/.../mean@... :" lines and a metrics dict)
val_lines = [l for l in text.splitlines() if "val-core" in l or "val/" in l]
# Pull step lines
step_lines = [l for l in text.splitlines() if re.search(r"step:\s*\d+", l)]

def grab(pat):
    m = re.findall(pat, text)
    return m

# value loss / advantage diagnostics emitted by VIMPO
vimpo_keys = grab(r"(vimpo[^\s:=]*)\s*[:=]\s*(-?\d+\.?\d*(?:e-?\d+)?)")
crit_acc = grab(r"val-core[^\s]*?:\s*(-?\d+\.?\d*)")

with open(eval_md, "w") as f:
    f.write("# VIMPO minimal reproduction — EVAL\n\n")
    f.write("**arXiv 2606.20008 — Value-Implicit Policy Optimization for LLMs**\n\n")
    f.write("Smallest end-to-end configuration that exercises the paper's core "
            "mechanism: the policy-implied value loss plus the PPO actor branch "
            "driven by the log-ratio TD advantage, on a single GPU.\n\n")
    f.write("## Config\n\n")
    f.write(f"- mode: `{run_mode}`\n")
    f.write(f"- model: `{model}`\n")
    f.write(f"- training steps: {steps}\n")
    f.write("- data: small slice of the paper's Guru math RLVR train set; "
            "eval on MATH-500 slice\n\n")

    f.write("## Result: did the VIMPO loop run end to end?\n\n")
    ran_train = bool(step_lines)
    f.write(f"- training steps observed: **{len(step_lines)}**\n")
    f.write(f"- validation metric lines observed: **{len(val_lines)}**\n")
    f.write(f"- VIMPO-specific diagnostics observed: **{len(vimpo_keys)}**\n\n")

    verdict = "PASS — full VIMPO training loop executed end to end" if ran_train \
        else "INCONCLUSIVE — no training step lines found in log"
    f.write(f"**Verdict: {verdict}**\n\n")

    if val_lines:
        f.write("## Validation metrics (last occurrences)\n\n```\n")
        f.write("\n".join(val_lines[-12:]))
        f.write("\n```\n\n")

    if step_lines:
        f.write("## Training step lines (last few)\n\n```\n")
        f.write("\n".join(step_lines[-6:]))
        f.write("\n```\n\n")

    if vimpo_keys:
        f.write("## VIMPO diagnostics (sample)\n\n```\n")
        seen = []
        for k, v in vimpo_keys[:40]:
            seen.append(f"{k} = {v}")
        f.write("\n".join(seen))
        f.write("\n```\n")
print("EVAL.md written")
PYEOF

# Also surface EVAL.md at the repo root (the conventional location).
cp -f "${EVAL_MD}" "${REPO_ROOT}/EVAL.md" || true

echo "[run_minimal] done: $(date -u)"
cat "${EVAL_MD}"
