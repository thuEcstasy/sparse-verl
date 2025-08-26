from PIL import Image
import os

def combine_images(image_folder, output_path, rows=3, cols=5):
    """
    将指定文件夹中的15张图片组合成一张大图
    
    参数:
    image_folder: 包含输入图片的文件夹路径
    output_path: 输出图片的保存路径
    rows: 组合后的行数
    cols: 组合后的列数
    """
    
    # 获取文件夹中的所有图片文件
    image_files = ['model_layers_26_self_attn_v_proj_weight_singular_values_normalized.png',
                   'model_layers_26_self_attn_k_proj_weight_singular_values_normalized.png',
                   'model_layers_26_self_attn_q_proj_weight_singular_values_normalized.png',
                   'model_layers_26_self_attn_o_proj_weight_singular_values_normalized.png',
                   'model_layers_26_mlp_up_proj_weight_singular_values_normalized.png',
                   'model_layers_26_mlp_gate_proj_weight_singular_values_normalized.png',
                   'model_layers_26_mlp_down_proj_weight_singular_values_normalized.png',
                   'model_layers_0_self_attn_v_proj_weight_singular_values_normalized.png',
                   'model_layers_0_self_attn_k_proj_weight_singular_values_normalized.png',
                   'model_layers_0_self_attn_q_proj_weight_singular_values_normalized.png',
                   'model_layers_0_self_attn_o_proj_weight_singular_values_normalized.png',
                   'model_layers_0_mlp_up_proj_weight_singular_values_normalized.png',
                   'model_layers_0_mlp_gate_proj_weight_singular_values_normalized.png',
                   'model_layers_0_mlp_down_proj_weight_singular_values_normalized.png']
    
    # 确保有足够的图片
    if len(image_files) < rows * cols:
        print(f"错误: 需要 {rows*cols} 张图片，但只找到 {len(image_files)} 张")
        # return
    
    # 取前15张图片
    image_files = image_files[:rows*cols]
    
    # 打开第一张图片以获取尺寸
    first_image = Image.open(os.path.join(image_folder, image_files[0]))
    img_width, img_height = first_image.size
    
    # 创建新的大图
    new_image = Image.new('RGB', (cols * img_width, rows * img_height))
    
    # 将每张图片粘贴到大图上
    for index, image_file in enumerate(image_files):
        img = Image.open(os.path.join(image_folder, image_file))
        
        # 计算位置
        x = (index % cols) * img_width
        y = (index // cols) * img_height
        
        new_image.paste(img, (x, y))
    
    # 保存结果
    new_image.save(output_path)
    print(f"图片已成功组合并保存到: {output_path}")

# 使用示例
if __name__ == "__main__":
    # 设置参数
    image_folder = "/home/haizhonz/Zhaofeng/verl/scripts/plot/layer_diff_plots"  # 替换为你的图片文件夹路径
    output_path = "combined_image.png"    # 输出文件路径
    
    # 调用函数组合图片
    combine_images(image_folder, output_path, rows=3, cols=5)