# 角色替换映射
character_replacements = {
    'Robinson': 'a man, wearing tattered sailor clothes.',
    'Friday': 'a chimpanzee.',
    'Lin Zhen': 'A female programmer, physically and mentally exhausted.',
    'Zhang': 'a college student in a blue jacket'
}

# 存储所有的故事集合
stories = {
    0: [
        "In a traditional meditation hall, an elderly master sits cross-legged on a cushion...",
        "In a natural courtyard, a young woman sits upright on a wooden bench...",
        "In a modern apartment close-up, a businessman leans on a sofa...",
        "In an abstract consciousness space, a translucent figure hovers...",
        "In a neuroscience laboratory, a researcher points to a holographic brain map..."
    ],
    1: [
        "In a park, a dejected Zhang is sitting on a bench, looking downcast, recollecting his failed exam despite hard revision.",
        "Zhang, with a furrowed brow, is deep in self - doubting thoughts like 'Am I too incompetent? Will I never succeed?'",
        "Zhang, with a determined look, is starting his mindfulness meditation session, ready to focus.",
        "With eyes closed, Zhang is following the system's guidance for mindful breathing, a serene expression on his face.",
        "Zhang opens his eyes, looking much calmer, realizing that failure is a chance to learn, not a reflection of his worth."
    ],
    2: [],
    3: [],
    4: [],
    5: [],
    6: []
}

def replace_characters(prompts, enable_character_replacement=False):
    """
    替换提示中的角色名称
    
    Args:
        prompts: 提示列表
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
    """
    根据输入获取对应的故事
    
    Args:
        story_input: 可以是故事列表或故事编号
        
    Returns:
        tuple: (story_content, story_number)
    """
    # 如果输入的是列表，说明是自定义故事
    if isinstance(story_input, list):
        return story_input, "custom"
        
    # 如果输入的是已存在的故事列表，查找对应的编号
    for story_num, story_content in stories.items():
        if story_input == story_content:
            return story_content, story_num
            
    return None, None 