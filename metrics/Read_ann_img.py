from pycocotools.coco import COCO
import skimage.io as io
import matplotlib.pyplot as plt
import os
import json

# 本部分代码的意义在于后续为data_loader做准备

# 如果出现HTTP Error 429: Too Many Requests
# 在运行代码时，涉及到迭代过程中很多图片，程序执行到一半时停止执行并持续报错。显示的是系统文件错误。
# 点击Pycharm右下角Python 3.10 > interpreter Settings > 选择左边菜单栏Tools > Python Scientific > 取消勾选show plots in tool window并保存
def get_caps_imgs(folder_path, dataType):
    # 先将json文件打开，避免每次都遍历
    with open(annFile, 'r') as f:
        image_info = json.load(f)

        for name in os.listdir(folder_path):
            # 通过名称查询每一个图片对应的专属id
            for img in image_info['images']:
                if img['file_name'] == name:
                    image_id = img['id']
                    print(f"The ID of {name} is: {image_id}")

                    Img = io.imread("D:/LLM/COCO/{}/".format(dataType) + name)
                    plt.imshow(Img)
                    plt.show()
                    annIds = coco.getAnnIds(imgIds=image_id)
                    anns = coco.loadAnns(annIds)
                    coco.showAnns(anns)
                    break
                break
            else:
                print(f"{name} not found in the dataset.")


if __name__ == '__main__':
    # coco init
    # Attention：在captions的情况下，不是所有的函数都被定义了(例如，categories 是没有定义的)。
    dataDir = 'D:/LLM/COCO'
    dataType = 'train2017'
    annFile = '{}/annotations/captions_{}.json'.format(dataDir, dataType)
    coco = COCO(annFile)

    folder_path = '/Traffic/images'  # 在这里通过前期筛选好的交通数据直接展示，可以把运行速度变快一些 注意train和val分开了
    dataDir = 'D:/LLM/COCO'
    dataType = 'train2017'
    get_caps_imgs(folder_path=folder_path, dataType=dataType)
