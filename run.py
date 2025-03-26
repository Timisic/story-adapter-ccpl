import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipelineLegacy, \
    DDIMScheduler, AutoencoderKL
from PIL import Image
import numpy as np
from ip_adapter import StoryAdapterXL
import os
import random
import argparse
from stories import get_story, replace_characters, stories


parser = argparse.ArgumentParser()
parser.add_argument('--base_model_path', default=r"./RealVisXL_V4.0", type=str)
parser.add_argument('--image_encoder_path', type=str, default=r"./IP-Adapter/sdxl_models/image_encoder")
parser.add_argument('--ip_ckpt', default=r"./IP-Adapter/sdxl_models/ip-adapter_sdxl.bin", type=str)
parser.add_argument('--style', type=str, default='realistic', choices=["comic","film","realistic"])
parser.add_argument('--device', default="cuda", type=str)
parser.add_argument('--story', type=int, default=0, help='故事编号 (0-6), 或使用自定义故事列表')

args = parser.parse_args()

base_model_path = args.base_model_path
image_encoder_path = args.image_encoder_path
ip_ckpt = args.ip_ckpt
device = args.device
style = args.style

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)

# load SD pipeline
pipe = StableDiffusionXLPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    scheduler=noise_scheduler,
    feature_extractor=None,
    safety_checker=None
)

seed = random.randint(0, 100000)
print(seed)

# load story-adapter
storyadapter = StoryAdapterXL(pipe, image_encoder_path, ip_ckpt, device)

character=True
# 获取故事内容
if isinstance(args.story, int):
    if args.story in stories:
        story_content = stories[args.story]
        story_num = args.story
    else:
        raise ValueError(f"故事编号 {args.story} 不存在")
else:
    story_content = args.story
    story_num = "custom"

prompts = replace_characters(story_content, character)

def create_story_directories(story_num):
    """
    创建故事相关的目录结构
    
    Args:
        story_num: 故事编号
    
    Returns:
        dict: 包含所有相关目录路径的字典
    """
    # 基础目录路径
    base_dir = './story'
    story_dir = f'{base_dir}/story{story_num}'
    
    # 创建目录结构
    directories = {
        'story': story_dir,
        'initial_results': f'{story_dir}/results_xl',
    }
    
    # 创建迭代结果目录
    for i in range(1, 11):  # 假设最多10次迭代
        directories[f'iteration_{i}'] = f'{story_dir}/results_xl{i}'
    
    # 创建所有必要的目录
    for dir_path in directories.values():
        os.makedirs(dir_path, exist_ok=True)
        
    return directories

def generate_initial_images(prompts, story_dirs, seed, style):
    """
    生成初始图像
    
    Args:
        prompts: 提示文本列表
        story_dirs: 目录路径字典
        seed: 随机种子
        style: 生成风格
    
    Returns:
        list: 生成的图像列表（调整为256x256大小）
    """
    initial_images = []
    
    print(f"\n正在生成初始图像，保存至: {story_dirs['initial_results']}")
    # 为每个提示生成初始图像
    for i, text in enumerate(prompts):
        images = storyadapter.generate(
            pil_image=None,
            num_samples=1,
            num_inference_steps=50,
            seed=seed,
            prompt=text,
            scale=0.3,
            use_image=False,
            style=style
        )
        
        # 保存生成的图像
        output_path = f'{story_dirs["initial_results"]}/img_{i}.png'
        grid = image_grid(images, 1, 1)
        grid.save(output_path)
        print(f"已保存第 {i+1}/{len(prompts)} 张初始图像")
        
        resized_image = Image.open(output_path).resize((256, 256))
        initial_images.append(resized_image)
    
    return initial_images

def generate_iterative_images(prompts, initial_images, story_dirs, seed, style):
    """
    进行迭代图像生成
    
    Args:
        prompts: 提示文本列表
        initial_images: 初始图像列表
        story_dirs: 目录路径字典
        seed: 随机种子
        style: 生成风格
    """
    scales = np.linspace(0.3, 0.5, 10)
    current_images = initial_images
    
    for i, scale in enumerate(scales, 1):
        current_dir = story_dirs[f"iteration_{i}"]
        print(f"\n开始第 {i} 轮迭代生成，图像将保存至: {current_dir}")
        new_images = []
        
        for y, text in enumerate(prompts):
            images = storyadapter.generate(
                pil_image=current_images,
                num_samples=1,
                num_inference_steps=50,
                seed=seed,
                prompt=text,
                scale=scale,
                use_image=True,
                style=style
            )
            
            # 保存生成的图像
            output_path = f'{current_dir}/img_{y}.png'
            grid = image_grid(images, 1, 1)
            grid.save(output_path)
            print(f"已保存第 {y+1}/{len(prompts)} 张图像")
            
            new_images.append(images[0].resize((256, 256)))
        
        print(f"第 {i} 轮迭代完成")
        current_images = new_images


story_dirs = create_story_directories(story_num)
initial_images = generate_initial_images(prompts, story_dirs, seed, style)
generate_iterative_images(prompts, initial_images, story_dirs, seed, style)
