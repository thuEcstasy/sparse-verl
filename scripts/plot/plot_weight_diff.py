import os
from safetensors.torch import load_file
import torch
import matplotlib.pyplot as plt
from transformers import AutoModel

# 路径和加载模型
sparse_state_dict = load_file("/home/haizhonz/Zhaofeng/verl/checkpoints/sparse-verl/Qwen2-1.5B-block-topk-sparse_rollout/global_step_200/model.safetensors")
dense_state_dict = load_file("/home/haizhonz/Zhaofeng/verl/checkpoints/sparse-verl/Qwen2-1.5B-dense_rollout/global_step_200/model.safetensors")
model3 = AutoModel.from_pretrained("Qwen/Qwen2.5-Math-1.5B", torch_dtype=torch.float32)
orig_state_dict = model3.state_dict()
print(orig_state_dict.keys())

output_dir = "./layer_diff_plots"
os.makedirs(output_dir, exist_ok=True)

def get_orig_key(safetensor_key):
    # 把safetensor的key映射成orig_state_dict的key
    # 示例：model.layers.4.self_attn.v_proj.weight -> layers.4.self_attn.v_proj.weight
    if safetensor_key.startswith("model."):
        return safetensor_key[len("model.") :]
    return safetensor_key

# 关注的所有权重后缀，包括self_attn和mlp的
proj_suffixes = [
    "q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
    "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
    "input_layernorm.weight", "post_attention_layernorm.weight"
]

# 找所有匹配层
layer_keys = []
# for key in sparse_state_dict.keys():
#     print(f"Processing key: {key}")
for suffix in proj_suffixes:
    layer_keys.extend([k for k in sparse_state_dict.keys() if suffix in k])

for key in layer_keys:
    orig_key = get_orig_key(key)
    if orig_key not in orig_state_dict:
        print(f"Warning: {orig_key} not found in original model. Skipping...")
        continue

    sparse_tensor = sparse_state_dict[key] - orig_state_dict[orig_key]
    dense_tensor = dense_state_dict[key] - orig_state_dict[orig_key]
    # 计算奇异值
    if sparse_tensor.dim() == 2:
        sparse_s = torch.linalg.svdvals(sparse_tensor)
        dense_s = torch.linalg.svdvals(dense_tensor)
        # normalized by the largest singular value
        sparse_s = sparse_s / sparse_s[0]
        dense_s = dense_s / dense_s[0]
        # 画图
        plt.figure(figsize=(8, 5))
        plt.plot(sparse_s.cpu().numpy(), label='Sparse diff singular values')
        plt.plot(dense_s.cpu().numpy(), label='Dense diff singular values')
        plt.yscale('log')
        plt.xlabel('Singular value index')
        plt.ylabel('Singular value (log scale)')
        plt.title(f'Singular values of difference tensors\n{key}')
        plt.legend()
        plt.grid(True)

        # 保存图像，文件名里替换点避免路径问题
        fname = key.replace(".", "_") + "_singular_values_normalized.png"
        plt.savefig(os.path.join(output_dir, fname))
        print(f"  Saved singular value plot to {fname}")
        plt.close()
        
    # 计算L2范数
    sparse_l2 = torch.norm(sparse_tensor).item()
    dense_l2 = torch.norm(dense_tensor).item()
    sparse_max = torch.max(torch.abs(sparse_tensor)).item()
    dense_max = torch.max(torch.abs(dense_tensor)).item()

    print(f"Layer {key}:")
    print(f"  L2 norm sparse diff: {sparse_l2:.6f}")
    print(f"  L2 norm dense diff:  {dense_l2:.6f}")

    print(f"  Max abs sparse diff: {sparse_max:.6f}")
    print(f"  Max abs dense diff:  {dense_max:.6f}")
        
    # compute cosine similarity between two tensors (flattened)
    sparse_flat = sparse_tensor.flatten()
    dense_flat = dense_tensor.flatten()
    print(sparse_flat, dense_flat)
    # print non-zero counts ratio
    sparse_nonzero = torch.count_nonzero(sparse_flat).item()
    dense_nonzero = torch.count_nonzero(dense_flat).item()
    total_elements = sparse_flat.numel()
    print(f"  Sparse diff non-zero count: {sparse_nonzero}/{total_elements} ({sparse_nonzero/total_elements:.4f})")
    print(f"  Dense diff non-zero count:  {dense_nonzero}/{total_elements} ({dense_nonzero/total_elements:.4f})")
    cos_sim = torch.nn.functional.cosine_similarity(sparse_flat, dense_flat, dim=0).item()
    print(f"  Cosine similarity between sparse and dense diff: {cos_sim:.6f}")
    
    rand_sparse_flat = torch.rand_like(sparse_flat)
    rand_dense_flat = torch.rand_like(dense_flat)

    rand_cos_sim = torch.nn.functional.cosine_similarity(rand_sparse_flat, rand_dense_flat, dim=0).item()
    print(f"  Cosine similarity between random tensors: {rand_cos_sim:.6f}")
