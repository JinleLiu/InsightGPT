import json
from tqdm import tqdm
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType,
    get_default_template_type, inference_stream
)
from swift.utils import seed_everything
import torch

model_type = ModelType.minicpm_v_v2_5_chat
template_type = get_default_template_type(model_type)
print(f'template_type: {template_type}')

model, tokenizer = get_model_tokenizer(model_type, torch.bfloat16,
                                       model_kwargs={'device_map': 'auto'})
model.generation_config.max_new_tokens = 256
template = get_template(template_type, tokenizer)
seed_everything(42)

# 在这里写一个循环，读取图片和问题
# source = [r"/home/ls/Use_to_train/eval/image_captioning.json", r"/home/ls/Use_to_train/eval/object_attributes_recognition.json", r"/home/ls/Use_to_train/eval/spatial_reasoning.json", r"/home/ls/Use_to_train/eval/template_recognition.json", r"/home/ls/Use_to_train/eval/traffic_violation_recognition.json", r"/home/ls/Use_to_train/eval/visual_reasoning.json"]
#
# target_file = [r"/home/ls/Use_to_train/eval/MiniCPM-V25_ans/image_captioning_ans.json", r"/home/ls/Use_to_train/eval/MiniCPM-V25_ans/object_attributes_recognition_ans.json", r"/home/ls/Use_to_train/eval/MiniCPM-V25_ans/spatial_reasoning_ans.json", r"/home/ls/Use_to_train/eval/MiniCPM-V25_ans/template_recognition_ans.json", r"/home/ls/Use_to_train/eval/MiniCPM-V25_ans/traffic_violation_recognition_ans.json", r"/home/ls/Use_to_train/eval/MiniCPM-V25_ans/visual_reasoning_ans.json"]

source = [r"/home/ls/Use_to_train/eval/coco_image_captioning.json", r"/home/ls/Use_to_train/eval/coco_object_attributes_recognition.json"]

target_file = [r"/home/ls/Use_to_train/eval/MiniCPM-V25_ans/coco_image_captioning_ans.json", r"/home/ls/Use_to_train/eval/MiniCPM-V25_ans/coco_object_attributes_recognition.json"]

for q in range(0, 2):
    all_answer = []
    with open(source[q], 'r', encoding='utf-8') as f:
        content = json.load(f)
        num = 0
        for i in tqdm(content): # 测试完记得删掉！
            images = i['images']
            query = i['query']
            response, history = inference(model, template, query, images=images)
            dict = {"query":query, "response": response, "idx": num, "image":images}
            all_answer.append(dict)
            num += 1

    with open(target_file[q], 'w', encoding='utf-8') as f1:
        json.dump(all_answer, f1, indent=4)