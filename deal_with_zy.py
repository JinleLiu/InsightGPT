from PIL import Image
import pytesseract
import os
import re
import json
from tqdm import tqdm

# 设置Tesseract路径
# pytesseract.pytesseract.tesseract_cmd = r'D:\LLM\tesseract\Tesseract-OCR\tesseract.exe'  # 修改为你的Tesseract路径 使用文本识别的时候再用
# 遍历文件夹内的所有图片

# pre_prompt(for single image in concept enhance(0) and concreate images in reasoning(1))
Prompt_prefix = ["You are an AI visual assistant, and you are seeing a single image. Here's a question you have to answer based on the content of the image.",
                 "You are an AI visual assistant looking at an image. This image consists of four parts, first there is a close-up of the target vehicle in the lower left corner, then the images in the upper left corner, the upper right corner and the lower left corner show the target vehicle in motion. Here's a question for you to answer based on the content of the entire image."]
# tasks prompts
image_captioning = ["Give a comprehensive and detailed analysis of the required image.",
                    "Please provide a detailed description of the following image.",
                    "Explain the various elements that comprise the image.",
                    "Provide an in-depth description of the given image.",
                    "Offer a comprehensive analysis of the image."]

object_attributes_recognition = ["Characterize the vehicle depicted in the red box in the provided image.",
                                 "Please describe the state of the signal light in the provided image."]

text_recognition = ["Provide a description of the license plate number of the vehicle in the provided image.",
                    "Please write the license plate number of the vehicle as shown in the given image.",
                    "Please provide the license plate number of the vehicle depicted in the image."]

visual_reasoning = ["Please provide a detailed account of the weather conditions in the provided image.",
                    "Please describe the weather conditions as illustrated in the image.",
                    "Determine the time of day based on the provided image."
                    "Please infer the time of day from the content of the image."]

spatial_reasoning = ["Please provide a detailed description of the trajectory of the vehicle shown in the lower right image.",
                     "Please delineate the route traversed of the vehicle shown in the lower right image.",
                     "Please describe the trajectory of the vehicle shown in the lower right image."]

# traffic_violation_recognition = ["Please identify violations (if any) of the vehicle shown in the lower right image based on the trajectory of the vehicle in the image, road markings, and the status of signal lights.",
#                                  "Based on the trajectory of the vehicle shown in the lower right image, road markings, and signal status, check whether the vehicle has violated any traffic laws. "]

# 0 for violation 1 for not violation
traffic_violation_recognition = ["The target vehicle shown in the lower right corner of the image has {} violation, explain why the violation exists based on the trajectory of the target vehicle in the upper left, upper right, and lower right images, the road markings present, and the status of the signal lights.".format(violation_type),
                                 "The target vehicle shown in the lower right corner of the image has no violation, explain why the target vehicle doesn't have a violation based on the trajectory of the target vehicle in the upper left, upper right, and lower right images, the road markings present, and the status of the signal lights."]


def get_car_licence(filename, pattern):
    match = pattern.search(filename)
    if match:
        print(f"file_name：{filename}，car_license：{match.group(1)}")
        new_name = match.group(1)  # 添加原始文件扩展名
    else:
        print(f"file_name：{filename}，miss matching。")
    return new_name


def get_cut_images(old_folder_path, new_folder_path, filename, new_id):
    img_path = os.path.join(old_folder_path, filename)
    image = Image.open(img_path)
    width, height = image.size
    # 生成剩余部分的图片
    top_part = image.crop((0, 0, width, height // 2.3))  # 顶部区域
    middle_part = image.crop((0, height // 2, width, height // 1.08))  # 中间部分

    # 拼接剩余部分
    remaining_height = top_part.height + middle_part.height
    remaining_image = Image.new('RGB', (width, remaining_height))
    remaining_image.paste(top_part, (0, 0))
    remaining_image.paste(middle_part, (0, top_part.height))

    # 显示或保存结果
    remaining_image.show()
    remaining_image.save(os.path.join(new_folder_path, str(new_id) + ".jpg"))


"""
每次跑新的数据之前:
1. 更改图片地址folder_path
2. 更改图片编号 new_id
3. 更改字典中的违法行为及id
"""
if __name__ == '__main__':
    folder_path = 'D:\博士\杂七杂八\方法论\中裕数据\图片数据\豫A5MX66'
    pattern = re.compile(r'([\u4e00-\u9fff].{6})')
    num = 1
    new_folder_path = 'D:\博士\杂七杂八\方法论\中裕数据\图片数据\Test'
    json_file_path = 'D:\博士\杂七杂八\方法论\中裕数据\图片数据\image_info.json'
    # 用来存储所有图片信息的列表
    images_info = []
    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith(('.jpg', '.png')):
            # 设置图片编号，A表示正样本，B表示负样本，前面的是违法行为，后面的是数字
            new_id = "A1039_" + str(num)
            # 截取图片并存储
            get_cut_image(old_folder_path=folder_path, filename=filename, new_id=new_id, new_folder_path=new_folder_path)
            car_licence = get_car_licence(filename, pattern)
            num += 1
            # 创建一个字典存储图片信息
            # 记得针对不同的违法行为更改违法id和type
            image_info = {
                'image_id': new_id,
                'description_id': new_id + 'des',
                'description':'This is a test',
                'car_licence':car_licence,
                'offence_type_id':'1039',
                'offence_type':'机动车违反规定停放'
            }
            # 添加到列表中
            images_info.append(image_info)

        # 写入JSON文件
        with open(json_file_path, 'w') as json_file:
            json.dump(images_info, json_file, ensure_ascii=False, indent=4)
