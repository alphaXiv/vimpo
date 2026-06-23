#!/usr/bin/env bash
# Scaled VIMPO-vs-GRPO comparison for arXiv 2606.20008.
#
# This branch runs a single objective (RUN_MODE below is hardcoded per branch).
# The VIMPO and GRPO branches share an IDENTICAL config except for RUN_MODE,
# isolating the paper's claim: the VIMPO objective (policy-implied value loss +
# PPO actor branch on the log-ratio TD advantage) vs the GRPO baseline.
set -uo pipefail

# ===== the only thing that differs between the two comparison branches =====
RUN_MODE="grpo"   # "vimpo" or "grpo"
# ===========================================================================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

ART_DIR="${REPO_ROOT}/.openresearch/artifacts"
mkdir -p "${ART_DIR}"
EVAL_MD="${ART_DIR}/EVAL.md"
RUN_LOG="${ART_DIR}/run.log"
exec > >(tee -a "${RUN_LOG}") 2>&1

fail() {
    echo "[run] FAILED: $*"
    {
        echo "# VIMPO comparison (${RUN_MODE}) — FAILED"
        echo
        echo "Stage: $*"
    } > "${EVAL_MD}"
    exit 1
}

echo "=============================================="
echo "VIMPO vs GRPO comparison — arm: ${RUN_MODE}"
echo "start: $(date -u)"
echo "=============================================="
nvidia-smi || true

############################################
# 1. Deps (same stack proven by the smoke run)
############################################
apt-get update -y >/dev/null 2>&1 || true
apt-get install -y git wget build-essential libnuma-dev numactl >/dev/null 2>&1 || true

export HF_HUB_ENABLE_HF_TRANSFER=1
export PIP_ROOT_USER_ACTION=ignore
export TOKENIZERS_PARALLELISM=false
PY=python3

echo "[run] installing vllm (long step)"
${PY} -m pip install --upgrade pip >/dev/null 2>&1 || true
${PY} -m pip install "vllm==0.17.0" || fail "pip install vllm"

echo "[run] installing verl runtime deps"
${PY} -m pip install \
    "accelerate" "codetiming" "datasets" "dill" "hydra-core" \
    "numpy<2.0.0" "pandas" "peft" "pyarrow>=19.0.0" "pybind11" "pylatexenc" \
    "ray[default]>=2.41.0" "torchdata" "tensordict>=0.8.0,<=0.10.0,!=0.9.0" \
    "transformers" "wandb" "packaging>=20.0" "tensorboard" \
    "math-verify" "latex2sympy2_extended" "liger-kernel" "hf_transfer" \
    || fail "pip install verl deps"

echo "[run] installing flash-attn (prebuilt preferred, else source build)"
${PY} -m pip install flash-attn --no-build-isolation \
    || echo "[run] WARNING: flash-attn install failed; falling back to sdpa"

${PY} -m pip install --no-deps -e . || fail "pip install -e . (verl)"

if ${PY} -c "import flash_attn" >/dev/null 2>&1; then
    USE_REMOVE_PADDING=True; ATTN_IMPL=""
    echo "[run] flash-attn available"
else
    USE_REMOVE_PADDING=False; ATTN_IMPL="sdpa"
    echo "[run] flash-attn NOT available -> sdpa"
fi

############################################
# 2. Data: paper's Guru math RLVR train + MATH-500 eval
############################################
DATA_DIR="${REPO_ROOT}/data"
mkdir -p "${DATA_DIR}"
TRAIN_FILE="${DATA_DIR}/math__combined_54.4k.parquet"
MATH500_FILE="${DATA_DIR}/math__math_500.parquet"
GURU="https://huggingface.co/datasets/LLM360/guru-RL-92k/resolve/main"
[ -f "${TRAIN_FILE}" ]  || wget -q -O "${TRAIN_FILE}"  "${GURU}/train/math__combined_54.4k.parquet"   || fail "download train"
[ -f "${MATH500_FILE}" ] || wget -q -O "${MATH500_FILE}" "${GURU}/offline_eval/math__math_500.parquet" || fail "download math500"
${PY} recipe/vimpo/data_preprocess/filter_test_dataset_keys.py --input_file "${MATH500_FILE}" || true

############################################
# 3. Scaled comparison config (identical across arms)
############################################
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-1.7B-Base}"

TRAIN_PROMPT_BSZ=32
N_RESP_PER_PROMPT=8
MAX_PROMPT_LENGTH=1024
MAX_RESPONSE_LENGTH=2048
TRAIN_MAX_SAMPLES=1024          # small slice of the 54.4k set
VAL_MAX_SAMPLES=200             # MATH-500 slice (same for both arms)
TOTAL_TRAINING_STEPS=30
TEST_FREQ=5                     # eval on MATH-500 every 5 steps
actor_ppo_max_token_len=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))

# W&B: stable names so the two arms are chartable side by side.
export WANDB_API_KEY="${WANDB_API_KEY:-}"
WANDB_PROJECT="vimpo-repro-compare"
EXP_NAME="compare-${RUN_MODE}-qwen3-1.7b"
if [ -n "${WANDB_API_KEY}" ]; then LOGGER='["console","wandb"]'; else LOGGER='["console"]'; fi

echo "[run] mode=${RUN_MODE} model=${MODEL_PATH} steps=${TOTAL_TRAINING_STEPS} logger=${LOGGER}"

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
    actor_rollout_ref.actor.optim.lr_warmup_steps=5 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${TRAIN_PROMPT_BSZ} \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.50 \
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
    trainer.logger="${LOGGER}" \
    trainer.project_name="${WANDB_PROJECT}" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.test_freq=${TEST_FREQ} \
    trainer.save_freq=-1 \
    trainer.total_epochs=10 \
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS} \
    trainer.default_local_dir="${REPO_ROOT}/ckpts/compare" \
    trainer.resume_mode=disable \
    "${hydra_attn[@]}"
TRAIN_RC=$?
set +x
[ ${TRAIN_RC} -ne 0 ] && fail "training exited code ${TRAIN_RC}"

############################################
# 4. EVAL.md: extract MATH-500 val accuracy trajectory
############################################
echo "[run] building EVAL.md"
${PY} - "$RUN_LOG" "$EVAL_MD" "$RUN_MODE" "$MODEL_PATH" "$TOTAL_TRAINING_STEPS" <<'PYEOF'
import re, sys
run_log, eval_md, run_mode, model, steps = sys.argv[1:6]
text = open(run_log, errors="replace").read()

# val accuracy at each eval: "val-core/math__math/acc/mean@1:<x>"
accs = re.findall(r"val-core/math__math/acc/mean@1:([0-9.]+)", text)
accs = [float(a) for a in accs]
# step number paired with the metric line
step_acc = []
for m in re.finditer(r"step:(\d+)[^\n]*?val-core/math__math/acc/mean@1:([0-9.]+)", text):
    step_acc.append((int(m.group(1)), float(m.group(2))))
# val-before-train (step 0) is logged separately
m0 = re.search(r"step:0 - val-aux/math__math/reward/mean@1:[0-9.]+[^\n]*?val-core/math__math/acc/mean@1:([0-9.]+)", text)

base_acc = accs[0] if accs else None
final_acc = accs[-1] if accs else None
best_acc = max(accs) if accs else None

with open(eval_md, "w") as f:
    f.write(f"# VIMPO comparison — arm: {run_mode}\n\n")
    f.write("arXiv 2606.20008. Identical config across the VIMPO and GRPO arms; "
            "only the training objective differs.\n\n")
    f.write("## Config\n\n")
    f.write(f"- mode: `{run_mode}`\n- model: `{model}`\n- training steps: {steps}\n")
    f.write("- data: 1024-sample slice of Guru math train; MATH-500 (200) eval every 5 steps\n\n")
    f.write("## MATH-500 validation accuracy (avg@1)\n\n")
    f.write(f"- baseline (step 0): **{base_acc}**\n")
    f.write(f"- best over training: **{best_acc}**\n")
    f.write(f"- final: **{final_acc}**\n")
    if base_acc is not None and best_acc is not None:
        f.write(f"- best improvement over baseline: **{best_acc - base_acc:+.4f}**\n")
    f.write("\n## Full eval trajectory (step, acc)\n\n```\n")
    for s, a in step_acc:
        f.write(f"step {s:>3}  acc@1 = {a:.4f}\n")
    f.write("```\n")
print("EVAL.md written; accs:", accs)
PYEOF

cp -f "${EVAL_MD}" "${REPO_ROOT}/EVAL.md" || true
echo "[run] done: $(date -u)"
cat "${EVAL_MD}"
