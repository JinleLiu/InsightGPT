import os
import json
from PIL import Image
import random
from tqdm import tqdm
import re

# 先把nb的车牌数据洗出来吧
# template_data = []
# with open(fr'D:\LLM\Datasets\Use_to_train\Swift\swift\nb2swift_template.json', 'r', encoding='utf-8') as f:
#     random_numbers = random.sample(range(0, 1197), 200)
#     raw_data = json.load(f)
#     for i in random_numbers:
#         eval_data = raw_data[i]
#         template_data.append(eval_data)
# target_file = fr'D:\LLM\Datasets\Use_to_train\Swift\swift\swift_244\eval\template_recognition.json'
# with open(target_file, 'w', encoding='utf-8') as f1:
#     json.dump(template_data, f1, indent=4)

# 然后洗出来Image Caption，Object Attribute Recognition， Visual Reasoning，Spatial Reason，Traffic Violation Recognition

def get_categories_data():
    image_captioning_list = []
    object_attributes_recognition_list = []
    visual_reasoning_list = []
    spatial_reasoning_list = []
    traffic_violation_recognition = []
    categories_list = [image_captioning_list, object_attributes_recognition_list, visual_reasoning_list, spatial_reasoning_list, traffic_violation_recognition]
    categories = ["image_captioning", "object_attributes_recognition", "visual_reasoning", "spatial_reasoning", "traffic_violation_recognition"]
    all = [json.loads(q) for q in open(r'D:\LLM\Datasets\Use_to_train\nb_data.json', 'r', encoding='utf-8')]
    for i in all:
        if i['violation_type'] == "ambiguous" or i["image"] == '3202.png':
            pass
        elif i['category'] == "image_captioning":
            query = i["prompt"]
            response = i["text"]
            images = "/home/ls/Use_to_train/Theimages/" + i["image"]
            new_data = {"query": query, "response": response, "images":[images]}
            image_captioning_list.append(new_data)
        elif i['category'] == "object_attributes_recognition":
            query = i["prompt"]
            response = i["text"]
            images = "/home/ls/Use_to_train/Theimages/" + i["image"]
            new_data = {"query": query, "response": response, "images":[images]}
            object_attributes_recognition_list.append(new_data)
        elif i['category'] == "visual_reasoning":
            query = i["prompt"]
            response = i["text"]
            images = "/home/ls/Use_to_train/Theimages/" + i["image"]
            new_data = {"query": query, "response": response, "images":[images]}
            visual_reasoning_list.append(new_data)
        elif i['category'] == "spatial_reasoning":
            query = i["prompt"]
            response = i["text"]
            images = "/home/ls/Use_to_train/Theimages/" + i["image"]
            new_data = {"query": query, "response": response, "images":[images]}
            spatial_reasoning_list.append(new_data)
        elif i['category'] == "traffic_violation_recognition":
            query = "You are an AI visual assistant looking at an image. This image consists of four parts, first there is a close-up of the target vehicle in the lower right corner, then the images in the upper left corner, the upper right corner and the lower left corner show the target vehicle in motion. Here's a question for you to answer based on the content of the entire image.The target vehicle shown in the lower right corner of the image has a potential violation, explain why the violation exists based on the trajectory of the target vehicle in the upper left, upper right, and lower right images, the road markings present, and the status of the signal lights."
            response = i["text"]
            images = "/home/ls/Use_to_train/Theimages/" + i["image"]
            new_data = {"query": query, "response": response, "images":[images]}
            traffic_violation_recognition.append(new_data)
        else:
            pass

    for i in range(0,5):
        target_file = fr"D:\LLM\Datasets\Use_to_train\Swift\swift\swift_244\categories\{categories[i]}.json".format()
        with open(target_file, 'w', encoding='utf-8') as f1:
            json.dump(categories_list[i], f1, indent=4)


def get_eval_data():
    categories = ["image_captioning", "object_attributes_recognition", "visual_reasoning", "spatial_reasoning", "traffic_violation_recognition"]
    for num in range(0,5):
        template_data = []
        source_file = fr"D:\LLM\Datasets\Use_to_train\Swift\swift\swift_244\categories\{categories[num]}.json".format()
        with open(source_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            random_numbers = random.sample(range(0, len(raw_data)), 200)
            for i in random_numbers:
                eval_data = raw_data[i]
                template_data.append(eval_data)
        target_file = fr'D:\LLM\Datasets\Use_to_train\Swift\swift\swift_244\eval\{categories[num]}.json'.format()
        with open(target_file, 'w', encoding='utf-8') as f1:
            json.dump(template_data, f1, indent=4)

def fix_template():
    file_name = fr'D:\LLM\Datasets\Use_to_train\Swift\swift\swift_244\eval\template_recognition.json'
    new_data = []
    with open(file_name, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
        for i in raw_data:
            i['query'] = i['query'] + ' Just output the license plate number.'
            new_data.append(i)
    with open(file_name, 'w', encoding='utf-8') as f1:
        json.dump(new_data, f1, indent=4)

def get_attribute_data():
    object_attributes_recognition_list = [] # 存放所有object_attributes_recognition数据
    all = [json.loads(q) for q in open(r'D:\LLM\Datasets\Use_to_train\nb_data.json', 'r', encoding='utf-8')]
    for i in all:
        if i['category'] == "object_attributes_recognition":
            query = i["prompt"]
            response = i["text"]
            images = "/home/ls/Use_to_train/Theimages/" + i["image"]
            new_data = {"query": query, "response": response, "images":[images]}
            object_attributes_recognition_list.append(new_data)

    traffic_light = [] # 存放所有object_attributes_recognition中的信号灯识别数据
    for item in object_attributes_recognition_list:
        if item['query'] == "You are an AI visual assistant, and you are seeing a single image. Here's a question you have to answer based on the content of the image.Please describe the state of the signal light in the provided image.":
            traffic_light.append(item)
        else:
            pass

    template_data = [] # 存放200条信号灯数据
    random_numbers = random.sample(range(0, len(traffic_light)), 200)
    for num in random_numbers:
        eval_data = traffic_light[num]
        template_data.append(eval_data)
    target_file = fr'D:\LLM\Datasets\Use_to_train\Swift\swift\swift_244\eval\object_attributes_recognition.json'.format()
    with open(target_file, 'w', encoding='utf-8') as f1:
        json.dump(template_data, f1, indent=4)



if __name__ == '__main__':
    get_attribute_data()
    # fix_template()
    # get_categories_data()
    # get_eval_data()