srun --partition=HGPU --gres=gpu:1 --time=48:00:00 --pty bash
srun --partition=defq --gres=gpu:1 --nodelist=node-gpu02 --time=72:00:00 --pty bash
srun --partition=HGPU --gres=gpu:1 --nodelist=node-gpu03 --time=47:00:00 --pty bash
srun --partition=defq --gres=gpu:1 --nodelist=cyh-c0-gpu002 --time=48:00:00 --pty bash

python3 -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /home/haizhonz/Zhaofeng/verl/scripts/checkpoints/sparse-verl/H200_deepscaleR-dense_rollout_new/global_step_640/actor \
    --target_dir /home/haizhonz/Zhaofeng/verl/scripts/checkpoints/sparse-verl/H200_deepscaleR-dense_rollout_new/global_step_640/actor/huggingface

python3 -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /home/haizhonz/Zhaofeng/verl/scripts/checkpoints/sparse-verl/H200_deepscaleR-sparse_TIS_32_rollout_new/global_step_460/actor \
    --target_dir /home/haizhonz/Zhaofeng/verl/scripts/checkpoints/sparse-verl/H200_deepscaleR-sparse_TIS_32_rollout_new/global_step_460/actor/huggingface

python3 -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /home/haizhonz/Zhaofeng/verl/scripts/checkpoints/sparse-verl/H200_deepscaleR-sparse_TIS_16_rollout_new/global_step_360/actor \
    --target_dir /home/haizhonz/Zhaofeng/verl/scripts/checkpoints/sparse-verl/H200_deepscaleR-sparse_TIS_16_rollout_new/global_step_360/actor/huggingface



## For TIS + {GSPO_clipped, SPEC} Ablation Study:

### rollout = 16, clip_ratio = 0.05, ctx = 16k, GSPO+TIS

python3 -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /home/haizhonz/Zhaofeng/verl/scripts/checkpoints/sparse-verl/H100_deepscaleR-16kctx_sparse_TIS_rollout16_GSPO1_clipped95/global_step_320/actor \
    --target_dir /home/haizhonz/Zhaofeng/verl/scripts/checkpoints/sparse-verl/H100_deepscaleR-16kctx_sparse_TIS_rollout16_GSPO1_clipped95/global_step_320/actor/huggingface



### rollout = 8, clip_ratio = 0.05, ctx = 16k, GSPO+TIS

python3 -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /home/haizhonz/Zhaofeng/verl/scripts/checkpoints/sparse-verl/H100_deepscaleR-16kctx_sparse_TIS_rollout8_GSPO1_clipped95/global_step_320/actor \
    --target_dir /home/haizhonz/Zhaofeng/verl/scripts/checkpoints/sparse-verl/H100_deepscaleR-16kctx_sparse_TIS_rollout8_GSPO1_clipped95/global_step_320/actor/huggingface


### rollout = 8, clip_ratio = 0.03, ctx = 16k, GSPO+TIS

python3 -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /home/haizhonz/Zhaofeng/verl/scripts/checkpoints/sparse-verl/H100_deepscaleR-16kctx_sparse_TIS_rollout8_GSPO1_clipped/global_step_320/actor \
    --target_dir /home/haizhonz/Zhaofeng/verl/scripts/checkpoints/sparse-verl/H100_deepscaleR-16kctx_sparse_TIS_rollout8_GSPO1_clipped/global_step_320/actor/huggingface

### rollout = 8, dense

python3 -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /home/haizhonz/Zhaofeng/verl/scripts/checkpoints/sparse-verl/H100_deepscaleR-ctx16k_dense_rollout8/global_step_320/actor \
    --target_dir /home/haizhonz/Zhaofeng/verl/scripts/checkpoints/sparse-verl/H100_deepscaleR-ctx16k_dense_rollout8/global_step_320/actor/huggingface

### rollout = 8, TIS

python3 -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /home/haizhonz/Zhaofeng/verl/scripts/checkpoints/sparse-verl/H100_deepscaleR-16kctx_sparse_TIS_rollout8/global_step_260/actor \
    --target_dir /home/haizhonz/Zhaofeng/verl/scripts/checkpoints/sparse-verl/H100_deepscaleR-16kctx_sparse_TIS_rollout8/global_step_260/actor/huggingface

### dense, grpo, TIS

python3 -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /home/haizhonz/Zhaofeng/verl/scripts/checkpoints/sparse-verl/H100_deepscaleR-ctx16k_dense_rollout8_TIS_GSPO/global_step_40/actor \
    --target_dir /home/haizhonz/Zhaofeng/verl/scripts/checkpoints/sparse-verl/H100_deepscaleR-ctx16k_dense_rollout8_TIS_GSPO/global_step_40/actor/huggingface

