import os
import re
from typing import List, Tuple, Optional, Dict
import logging
import cv2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
import fitz
import shapely.geometry as sg
from shapely.geometry.base import BaseGeometry
from shapely.validation import explain_validity
import concurrent.futures

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import numpy as np
import fitz
import logging
import os
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)
from PIL import Image

from rapid_layout import RapidLayout, VisLayout

layout_engine = RapidLayout(conf_thres=0.5, model_type="pp_layout_cdla")

model = Qwen2VLForConditionalGeneration.from_pretrained("/home/user/Downloads/Qwen2-VL-7B-Instruct/",
                                                        torch_dtype=torch.bfloat16,
                                                        device_map="auto")

min_pixels = 256*28*28
max_pixels = 1280*28*28
processor = AutoProcessor.from_pretrained("/home/user/Downloads/Qwen2-VL-7B-Instruct/", 
                                          min_pixels=min_pixels, 
                                          max_pixels=max_pixels)

# This Default Prompt Using Chinese and could be changed to other languages.
DEFAULT_PROMPT = """使用markdown语法，将图片中识别到的文字转换为markdown格式输出。你必须做到：
1. 输出和使用识别到的图片的相同的语言，例如，识别到英语的字段，输出的内容必须是英语。
2. 不要解释和输出无关的文字，直接输出图片中的内容。例如，严禁输出 “以下是我根据图片内容生成的markdown文本：”这样的例子，而是应该直接输出markdown。
3. 内容不要包含在```markdown ```中、段落公式使用 $$ $$ 的形式、行内公式使用 $ $ 的形式、忽略掉长直线、忽略掉页码。
再次强调，不要解释和输出无关的文字，直接输出图片中的内容。
"""
DEFAULT_RECT_PROMPT = """图片中用带颜色的矩形框和名称(%s)标注出了一些区域。如果区域是表格或者图片，使用 ![]() 的形式插入到输出内容中，否则直接输出文字内容。
"""
DEFAULT_ROLE_PROMPT = """你是一个PDF文档解析器，使用markdown和latex语法输出图片的内容。
"""

def _parse_pdf_to_images(pdf_path: str, output_dir: str = './output') -> List[Tuple[str, List[str]]]:
    image_infos = []
    pdf_document = fitz.open(pdf_path)
    for page_index, page in enumerate(pdf_document):
        rect_images = []
        logging.info(f'parse page: {page_index}')
        # 保存页面为图片
        pix = page.get_pixmap(matrix=fitz.Matrix(4, 4))
        pix = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
        boxes, scores, class_names, elapse = layout_engine(pix)
        for index, (class_name, box) in enumerate(zip(class_names, boxes)):
            if class_name == 'figure' or class_name == 'table':
                name = f'{page_index}_{index}.png'
                sub_pix = pix.crop(box)
                sub_pix.save(os.path.join(output_dir, name))
                rect_images.append(name)

        boxes_ = []
        scores_ = []
        class_names_ = []
        for i, (class_name, box, score) in enumerate(zip(class_names, boxes, scores)):
            if class_name == 'figure' or class_name == 'table':
                boxes_.append(box)
                scores_.append(score)
                class_name = f'{page_index}_{i}.png'
                class_names_.append(class_name)
                
        page_image = os.path.join(output_dir, f'{page_index}.png')
        pix = np.array(pix)
        pix = cv2.cvtColor(pix, cv2.COLOR_RGB2BGR)
        print(boxes_, scores_, class_names_)
        ploted_img = VisLayout.draw_detections(pix, boxes_, scores_, class_names_)
        if ploted_img is not None:
            cv2.imwrite(page_image, ploted_img)
        # ploted_img.save(page_image)
        image_infos.append((page_image, rect_images))
    pdf_document.close()
    return image_infos


def _gpt_parse_images(
        image_infos: List[Tuple[str, List[str]]],
        prompt_dict: Optional[Dict] = None,
        output_dir: str = './',
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        # model: str = 'gpt-4o',
        verbose: bool = False,
        gpt_worker: int = 1,
        **args
) -> str:
    """
    Parse images to markdown content.
    """

    if isinstance(prompt_dict, dict) and 'prompt' in prompt_dict:
        prompt = prompt_dict['prompt']
        logging.info("prompt is provided, using user prompt.")
    else:
        prompt = DEFAULT_PROMPT
        logging.info("prompt is not provided, using default prompt.")
    if isinstance(prompt_dict, dict) and 'rect_prompt' in prompt_dict:
        rect_prompt = prompt_dict['rect_prompt']
        logging.info("rect_prompt is provided, using user prompt.")
    else:
        rect_prompt = DEFAULT_RECT_PROMPT
        logging.info("rect_prompt is not provided, using default prompt.")
    if isinstance(prompt_dict, dict) and 'role_prompt' in prompt_dict:
        role_prompt = prompt_dict['role_prompt']
        logging.info("role_prompt is provided, using user prompt.")
    else:
        role_prompt = DEFAULT_ROLE_PROMPT
        logging.info("role_prompt is not provided, using default prompt.")

    def _process_page(index: int, image_info: Tuple[str, List[str]]) -> Tuple[int, str]:
        logging.info(f'gpt parse page: {index}')

        # agent = Agent(role=role_prompt, api_key=api_key, base_url=base_url, disable_python_run=True, model=model, **args)
        page_image, rect_images = image_info
        local_prompt = prompt
        local_prompt = role_prompt + local_prompt
        if rect_images:
            local_prompt += rect_prompt + ', '.join(rect_images)
        # content = agent.run([local_prompt, {'image': page_image}], display=verbose)
        messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": page_image,
            },
            {"type": "text", "text": local_prompt},
        ],
    }
]
        text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
        print(text)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=2000, num_beams=1)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return index, output_text

    contents = [None] * len(image_infos)
    with concurrent.futures.ThreadPoolExecutor(max_workers=gpt_worker) as executor:
        futures = [executor.submit(_process_page, index, image_info) for index, image_info in enumerate(image_infos)]
        for future in concurrent.futures.as_completed(futures):
            index, content = future.result()
            content = content[0]
            print(content)

            # 在某些情况下大模型还是会输出 ```markdown ```字符串
            if '```markdown' in content:
                content = content.replace('```markdown\n', '')
                last_backticks_pos = content.rfind('```')
                if last_backticks_pos != -1:
                    content = content[:last_backticks_pos] + content[last_backticks_pos + 3:]

            contents[index] = content

    output_path = os.path.join(output_dir, 'output.md')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(contents))

    return '\n\n'.join(contents)


def parse_pdf(
        pdf_path: str,
        output_dir: str = './',
        prompt: Optional[Dict] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = 'gpt-4o',
        verbose: bool = False,
        gpt_worker: int = 1,
        **args
) -> Tuple[str, List[str]]:
    """
    Parse a PDF file to a markdown file.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_infos = _parse_pdf_to_images(pdf_path, output_dir=output_dir)
    print(image_infos)
    content = _gpt_parse_images(
        image_infos=image_infos,
        output_dir=output_dir,
        prompt_dict=prompt,
        api_key=api_key,
        base_url=base_url,
        model=model,
        verbose=verbose,
        gpt_worker=gpt_worker,
        **args
    )

    all_rect_images = []
    # remove all rect images
    if not verbose:
        for page_image, rect_images in image_infos:
            if os.path.exists(page_image):
                os.remove(page_image)
            all_rect_images.extend(rect_images)
    return content, all_rect_images


result = parse_pdf(
    pdf_path='/home/user/wyf/test.pdf',
    output_dir="./output",
    verbose=True,
    gpt_worker=1
)