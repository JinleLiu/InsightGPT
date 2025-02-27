import os
from PIL import Image
from tqdm import tqdm


def delete_corrupted_images(root_folder):
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in tqdm(filenames):
            if filename.endswith('.png'):
                file_path = os.path.join(dirpath, filename)
                try:
                    img = Image.open(file_path)  # 尝试打开图片
                    img.verify()  # 验证图片，如果图片损坏，这将引发异常
                except (IOError, SyntaxError) as e:
                    print('Bad file:', file_path)  # 打印出损坏的图片的路径

if __name__ == '__main__':
    root_folder = r'D:\LLM\Datasets\Use_to_train\Theimages1'
    delete_corrupted_images(root_folder)
