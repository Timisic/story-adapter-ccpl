import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipelineLegacy, \
    DDIMScheduler, AutoencoderKL
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from ip_adapter import StoryAdapterXL
import os
import random
import argparse
from stories import get_story, replace_characters


parser = argparse.ArgumentParser()
parser.add_argument('--base_model_path', default=r"./RealVisXL_V4.0", type=str)
parser.add_argument('--image_encoder_path', type=str, default=r"./IP-Adapter/sdxl_models/image_encoder")
parser.add_argument('--ip_ckpt', default=r"./IP-Adapter/sdxl_models/ip-adapter_sdxl.bin", type=str)
parser.add_argument('--style', type=str, default='film', choices=["comic","film","realistic"])
parser.add_argument('--device', default="cuda", type=str)
parser.add_argument('--story', type=int, default=0, help='故事编号，或使用自定义故事列表')
parser.add_argument('--use_annotations', action='store_true', help='是否在图像上添加文字注释')


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

character = True
# 获取故事内容
story_content, story_num, character_replacements, annotations = get_story(args.story)
if story_content is None:
    raise ValueError(f"故事编号 {args.story} 不存在")


prompts = replace_characters(story_content, character_replacements, character)

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
    for i in range(1, 9):
        directories[f'iteration_{i}'] = f'{story_dir}/results_xl{i}'
    
    # 创建所有必要的目录
    for dir_path in directories.values():
        os.makedirs(dir_path, exist_ok=True)
        
    return directories

def calculate_grid_size(num_images):
    """计算最优的网格大小，避免过多留白"""
    if num_images <= 4:
        return 2, 2
    cols = int(np.ceil(np.sqrt(num_images)))
    rows = int(np.ceil(num_images / cols))
    if rows * cols - num_images >= cols:
        rows -= 1
    return rows, cols

def add_text_to_image(image, text):
    """在图像上添加文字注释，类似字幕效果"""
    draw = ImageDraw.Draw(image)
    # 计算合适的字体大小（基于图像高度）
    font_size = int(image.height * 0.05)  # 图像高度的5%
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc", font_size)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf", font_size)
        except:
            font = ImageFont.load_default()
            font_size = 24

    # 计算每行最大字符数
    max_chars_per_line = int(image.width / (font_size * 0.8))  # 0.7是一个经验值，可以根据需要调整
    
    # 文本分行
    words = text.split()
    lines = []
    current_line = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) + 1 <= max_chars_per_line:
            current_line.append(word)
            current_length += len(word) + 1
        else:
            lines.append(' '.join(current_line))
            current_line = [word]
            current_length = len(word)
    
    if current_line:
        lines.append(' '.join(current_line))
    
    # 如果只有一行，直接使用原文本
    if not lines:
        lines = [text]
    
    # 计算所有文本的总高度
    line_spacing = int(font_size * 1.2)  # 行间距
    total_text_height = len(lines) * line_spacing
    
    # 计算起始Y坐标（从底部向上）
    y = image.height - total_text_height - int(image.height * 0.08)  # 距底部8%
    
    # 添加半透明黑色背景
    padding = int(font_size * 0.5)  # 文字周围的内边距
    background_box = [0, y - padding, image.width, y + total_text_height + padding]
    draw.rectangle(background_box, fill=(0, 0, 0, 80))  # 降低透明度（140 -> 更透明）
    
    # 绘制每一行文字
    for i, line in enumerate(lines):
        # 计算当前行的宽度和位置
        text_bbox = draw.textbbox((0, 0), line, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        x = (image.width - text_width) // 2
        current_y = y + i * line_spacing
        
        # 绘制文字
        draw.text((x, current_y), line, font=font, fill=(255, 255, 255))

    return image

def generate_initial_images(prompts, story_dirs, seed, style, annotations=None, use_annotations=False):
    initial_images = []
    
    print(f"\n正在生成初始图像，保存至: {story_dirs['initial_results']}")
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
        if use_annotations and annotations and i < len(annotations):
            grid = add_text_to_image(grid, annotations[i])
        grid.save(output_path)
        print(f"已保存第 {i+1}/{len(prompts)} 张初始图像")
        
        resized_image = Image.open(output_path).resize((256, 256))
        initial_images.append(resized_image)
    
    # 拼接组合图像
    num_images = len(initial_images)
    rows, cols = calculate_grid_size(num_images)
    
    # 只填充必要的空白图像
    needed_images = rows * cols
    while len(initial_images) < needed_images:
        blank_image = Image.new('RGB', (256, 256), 'white')
        initial_images.append(blank_image)
    
    combined = image_grid(initial_images[:needed_images], rows, cols)
    combined.save(f'{story_dirs["initial_results"]}/combined_figure.png')
    print(f"已保存组合图像至 {story_dirs['initial_results']}/combined_figure.png")
    
    return initial_images

def generate_iterative_images(prompts, initial_images, story_dirs, seed, style, annotations=None, use_annotations=False):
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
            if use_annotations and annotations and y < len(annotations):
                grid = add_text_to_image(grid, annotations[y])
            grid.save(output_path)
            print(f"已保存第 {y+1}/{len(prompts)} 张图像")
            
            new_images.append(images[0].resize((256, 256)))
        
        num_images = len(new_images)
        rows, cols = calculate_grid_size(num_images)
        
        needed_images = rows * cols
        while len(new_images) < needed_images:
            blank_image = Image.new('RGB', (256, 256), 'white')
            new_images.append(blank_image)
        
        combined = image_grid(new_images[:needed_images], rows, cols)
        combined.save(f'{current_dir}/combined_figure.png')
        print(f"已保存第 {i} 轮迭代的组合图像")
        
        print(f"第 {i} 轮迭代完成")
        current_images = new_images


story_dirs = create_story_directories(story_num)
initial_images = generate_initial_images(prompts, story_dirs, seed, style, annotations, args.use_annotations)
generate_iterative_images(prompts, initial_images, story_dirs, seed, style, annotations, args.use_annotations)
