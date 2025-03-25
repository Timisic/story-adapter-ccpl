import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipelineLegacy, \
    DDIMScheduler, AutoencoderKL
from PIL import Image
import numpy as np
from ip_adapter import StoryAdapterXL
import os
import random
import argparse

# 使用英文prompt，注释加中文即可
story0 = [
    "In a traditional meditation hall, an elderly master sits cross-legged on a cushion, forming a sacred mudra with his hands on his knees with serene expression, while a calligraphy scroll reading 'The Path is Here and Now' hangs on the background wall.",
    
    "In a natural courtyard, a young woman sits upright on a wooden bench, eyes closed while adjusting her breathing rhythm with relaxed brows, as maple leaves float suspended near her shoulder.",
    
    "In a modern apartment close-up, a businessman leans on a sofa, his hand gently touching his rising and falling chest with a slight smile, while his phone displays a 'mindful breathing reminder' notification.",
    
    "In an abstract consciousness space, a translucent figure hovers as colorful thought bubbles are pierced one by one by silver rings of attention, the broken bubbles dissolving into starlight.",
    
    "In a neuroscience laboratory, a researcher points to a holographic brain map, where the prefrontal cortex and amygdala regions light up in different colors, with neural connections pulsing in golden light."
]

story1 = [
    "In a wide-angle shot of an office building late at night, Lin Zhen curls up in the blue light of the computer, with fingers spasmodically tapping on the keyboard, showing an expression of anxiety and suffocation, and there are stacked coffee cups and bottles of sleeping pills beside.",
    "In an empty shot of a bamboo forest in the morning mist, Lin Zhen stands on a creek stone following a Zen master, feeling the water flow with bare feet, looking confused and tentative, and there is a wooden carving of a breathing guide covered with moss nearby.",
    "From an upward shooting perspective of the iron net on the rooftop, Lin Zhen kneels and sits on the waterlogged ground, facing up to bear the scouring of the rainstorm, showing pain and struggle, and the anti-anxiety pills are washed away.",
    "From an overhead shot in an infinite mirror palace, seven Lin Zhens walk simultaneously, touching the mirror surface and triggering ripples, showing cognitive fragmentation, and a rusted key in hand reflects different lives.",
    "From a microscopic perspective in the inner brain universe, Lin Zhen suspends among the neural synapses, stirring the flames of the amygdala, showing astonishment and sudden enlightenment, and the burning anxiety letters turn into a chain of stars.",
    "In the dusk glow of the rehabilitation room, Lin Zhen dances a waltz with the holographic projection of phantom limb pain, actively leading the trajectory of the pain, with a complex feeling of sorrow and joy, and the electronic bandages peel off with the dance steps.",
    "In a full view of the same office, Lin Zhen pushes open the French windows, and the papers flutter in the wind like a flock of birds, showing a sense of clarity and freedom, and there is a withered wood potted plant with a sprout at the table corner."
] 


parser = argparse.ArgumentParser()
parser.add_argument('--base_model_path', default=r"./RealVisXL_V4.0", type=str)
parser.add_argument('--image_encoder_path', type=str, default=r"./IP-Adapter/sdxl_models/image_encoder")
parser.add_argument('--ip_ckpt', default=r"./IP-Adapter/sdxl_models/ip-adapter_sdxl.bin", type=str)
parser.add_argument('--style', type=str, default='realistic', choices=["comic","film","realistic"])
parser.add_argument('--device', default="cuda", type=str)
parser.add_argument('--story', default=story1, nargs='+', type=str)

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
fixing_prompts = []
for prompt in args.story:
    if character == True:
        if 'Robinson' in prompt:
            prompt = prompt.replace('Robinson', 'a man, wearing tattered sailor clothes.')
        if 'Friday' in prompt:
            prompt = prompt.replace('Friday', 'a chimpanzee.')
        if 'Lin Zhen' in prompt:
            prompt = prompt.replace('Lin Zhen', 'A female programmer, working in a high-pressure environment for a long time, physically and mentally exhausted. ')
    fixing_prompts.append(prompt)

prompts = fixing_prompts

# 确定使用的是哪个story
if args.story == story1:
    story_num = 1
elif args.story == story0:
    story_num = 0
elif args.story == story2:
    story_num = 2
elif args.story == story3:
    story_num = 3
elif args.story == story4:
    story_num = 4
elif args.story == story5:
    story_num = 5
elif args.story == story6:
    story_num = 6
else:
    story_num = "custom"

# 使用story_num创建对应的目录
os.makedirs(f'./story/story{story_num}', exist_ok=True)
os.makedirs(f'./story/story{story_num}/results_xl', exist_ok=True)

for i, text in enumerate(prompts):
    images = storyadapter.generate(pil_image=None, num_samples=1, num_inference_steps=50, seed=seed,
            prompt=text, scale=0.3, use_image=False, style=style)
    grid = image_grid(images, 1, 1)
    grid.save(f'./story/story{story_num}/results_xl/img_{i}.png')

images = []
for y in range(len(prompts)):
    image = Image.open(f'./story/story{story_num}/results_xl/img_{y}.png')
    image = image.resize((256, 256))
    images.append(image)

scales = np.linspace(0.3,0.5,10)
print(scales)

for i, scale in enumerate(scales):
    new_images = []
    os.makedirs(f'./story/story{story_num}/results_xl{i+1}', exist_ok=True)
    print(f'epoch:{i+1}')
    for y, text in enumerate(prompts):
        image = storyadapter.generate(pil_image=images, num_samples=1, num_inference_steps=50, seed=seed,
                                  prompt=text, scale=scale, use_image=True, style=style)
        new_images.append(image[0].resize((256, 256)))
        grid = image_grid(image, 1, 1)
        grid.save(f'./story/story{story_num}/results_xl{i+1}/img_{y}.png')
    images = new_images
