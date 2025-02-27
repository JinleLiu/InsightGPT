# -*- coding: utf-8 -*-
from pycocotools.coco import COCO
import os
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw
import skimage.io as io
import json
"""
出处：https://www.cnblogs.com/lhdb/p/13218084.html
在windows下安装pycocotools建议使用
"""
'''
 路径参数
 '''
#原coco数据集的路径
dataDir= 'D:\LLM\COCO'
#用于保存新生成的数据的路径
savepath = "D:/LLM/COCO/Traffic_val"
#只保存含有你需要的类别的图片的路径，最后没有用
#因为没必要，coco是按json中的信息读图，只要在json里做筛选就行了
img_save = savepath + 'images/'
#最后生产的json文件的保存路径
anno_save = savepath+'annotations/'
'''
数据集参数
'''
#coco有80类，这里写要提取部分类的名字
#如我只需要car、bus、truck这三类数据
classes_names = ['car','bus','truck','traffic light','stop sign','bicycle','motorcycle']
#要处理的数据集，比如val2017、train2017等
#不建议多个数据集在一个list中，我就跑崩了
#还是一次提取一个数据集安全点_(:3」∠❀)_
datasets_list=['train2017']

#生成保存路径，函数抄的(›´ω`‹ )
#if the dir is not exists,make it,else delete it
def mkr(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.mkdir(path)
    else:
        os.mkdir(path)

#获取并处理所有需要的json数据
def process_json_data(annFile):
    #获取COCO_json的数据
    coco = COCO(annFile)
    #拿到所有需要的图片数据的id
    classes_ids = coco.getCatIds(catNms = classes_names)
    #加载所有需要的类别信息
    classes_list = coco.loadCats(classes_ids)
    #取所有类别的并集的所有图片id
    #如果想要交集，不需要循环，直接把所有类别作为参数输入，即可得到所有类别都包含的图片
    imgIds_list = []
    for idx in classes_ids:
        imgidx = coco.getImgIds(catIds=idx)
        imgIds_list += imgidx
    #去除重复的图片
    imgIds_list = list(set(imgIds_list))
    #一次性获取所有图像的信息
    image_info_list = coco.loadImgs(imgIds_list)
    #获取图像中对应类别的分割信息,由catIds来指定
    annIds = coco.getAnnIds(imgIds = [], catIds = classes_ids, iscrowd=None)
    anns_list = coco.loadAnns(annIds)
    return classes_list,image_info_list,anns_list

#保存数据到json
def save_json_data(json_file,classes_list,image_info_list,anns_list):
    coco_sub = dict()
    coco_sub['info'] = dict()
    coco_sub['licenses'] = []
    coco_sub['images'] = []
    coco_sub['type'] = 'instances'
    coco_sub['annotations'] = []
    coco_sub['categories'] = []
    #以下非必须,为coco数据集的前缀信息
    coco_sub['info']['description'] = 'COCO 2017 sub Dataset'
    coco_sub['info']['url'] = 'https://www.cnblogs.com/lhdb/'
    coco_sub['info']['version'] = '1.0'
    coco_sub['info']['year'] = 2020
    coco_sub['info']['contributor'] = 'smh'
    coco_sub['info']['date_created'] = '2020-7-1 10:06'
    sub_license = dict()
    sub_license['url'] =  'https://www.cnblogs.com/lhdb/'
    sub_license['id'] = 1
    sub_license['name'] = 'Attribution-NonCommercial-ShareAlike License'
    coco_sub['licenses'].append(sub_license)
    #以下为必须插入信息,包括image、annotations、categories三个字段
    #插入image信息
    coco_sub['images'].extend(image_info_list)
    #插入annotation信息
    coco_sub['annotations'].extend(anns_list)
    #插入categories信息
    coco_sub['categories'].extend(classes_list)
    #自此所有该插入的数据就已经插入完毕啦
    #最后一步，保存数据
    json.dump(coco_sub, open(json_file, 'w'))

if __name__ == '__main__':
    mkr(img_save)
    mkr(anno_save)
    #按单个数据集进行处理
    for dataset in datasets_list:
        #获取要处理的json文件路径
        annFile='{}/annotations/instances_{}.json'.format(dataDir,dataset)
        #存储处理完成的json文件路径
        json_file = '{}/instances_{}_sub.json'.format(anno_save,dataset)
        #处理数据
        classes_list,image_info_list,anns_list = tqdm(process_json_data(annFile))
         #保存数据
        tqdm(save_json_data(json_file,classes_list,image_info_list,anns_list))
        print('instances_{}_sub.json saved ٩( ๑╹ ꇴ ╹)۶'.format(dataset))