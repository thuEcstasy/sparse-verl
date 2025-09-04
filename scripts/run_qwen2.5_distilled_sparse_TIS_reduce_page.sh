#!/bin/bash
#SBATCH --job-name=sparse-verl
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --nodelist=node-gpu02
#SBATCH --time=70:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

conda init
source ~/.bashrc
conda activate verl-sparse

export SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=True
export PYTHONPATH=/home/haizhonz/Zhaofeng/sglang/python:$PYTHONPATH
export CUDA_LAUNCH_BLOCKING=1
export IMP_RATIO_CAP=4
module load cuda12.4/toolkit/12.4.1
nvcc -V

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/home/haizhonz/Zhaofeng/verl/scripts/data/lighteval-math/train.parquet \
    data.val_files=/home/haizhonz/Zhaofeng/verl/scripts/data/math500/test_128.parquet \
    data.train_batch_size=16 \
    data.val_batch_size=64 \
    data.max_prompt_length=512 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend='flashinfer' \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.disable_cuda_graph=True \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.vortex_num_selected_pages=16 \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.page_size=16 \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.vortex_sparse_attention_algorithm='BLOCK_TOPK' \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.disable_overlap_schedule=True \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.enable_vortex_sparsity=True \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.vortex_page_reserved_bos=2 \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.vortex_page_reserved_eos=2 \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.vortex_layers_skip=[0,1] \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console, wandb'] \
    trainer.project_name='sparse-verl' \
    trainer.experiment_name='Deepseek_r1_distill_1.5B-block-topk-sparse+TIS_rollout' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.total_epochs=15 \
    trainer.resume_mode=disable \
    actor_rollout_ref.rollout.calculate_log_probs=True