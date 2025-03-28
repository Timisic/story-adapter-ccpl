import os
import json

def load_story_from_json(story_num):
    """从JSON文件加载故事内容
    
    Args:
        story_num: 故事编号
        
    Returns:
        dict: 包含故事内容、角色替换和注释的字典
    """
    story_path = os.path.join(os.path.dirname(__file__), 'stories', f'story{story_num}.json')
    if not os.path.exists(story_path):
        return None
        
    with open(story_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def replace_characters(prompts, character_replacements, enable_character_replacement=False):
    """替换提示中的角色名称
    
    Args:
        prompts: 提示列表
        character_replacements: 角色替换映射
        enable_character_replacement: 是否启用角色替换
        
    Returns:
        list: 处理后的提示列表
    """
    if not enable_character_replacement:
        return prompts
        
    fixed_prompts = []
    for prompt in prompts:
        fixed_prompt = prompt
        for character, replacement in character_replacements.items():
            if character in fixed_prompt:
                fixed_prompt = fixed_prompt.replace(character, replacement)
        fixed_prompts.append(fixed_prompt)
    
    return fixed_prompts

def get_story(story_input):
    """根据输入获取对应的故事
    
    Args:
        story_input: 可以是故事列表或故事编号
        
    Returns:
        tuple: (story_content, story_number, character_replacements, annotations)
    """
    # 如果输入的是列表，说明是自定义故事
    if isinstance(story_input, list):
        return story_input, "custom", {}, []
    
    # 尝试加载故事JSON文件
    story_data = load_story_from_json(story_input)
    if story_data:
        return (
            story_data['story'],
            story_input,
            story_data.get('character_replacements', {}),
            story_data.get('annotations', [])
        )
            
    return None, None, None, None