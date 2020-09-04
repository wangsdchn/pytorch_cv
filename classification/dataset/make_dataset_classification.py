import os
import random
from pathlib import Path


def is_image_file(imgname):
    IMG_EXTS = ['.jpg', '.jpeg', '.png']
    is_image_flag = True if os.path.splitext(imgname.lower())[1] in IMG_EXTS else False
    return is_image_flag


def make_train_test_txt(image_folder, train_txt, test_txt, test_ratio=0.1, check_label=False):
    sub_folders = [folder for folder in image_folder.glob('*') if folder.is_dir()]
    sub_folders.sort()
    label_dict = dict([[sub_folder.stem, label] for label, sub_folder in enumerate(sub_folders)])
    print(label_dict)
    if check_label:
        return
    all_train_data, all_test_data = [], []
    for label, sub_folder in enumerate(sub_folders):
        train_test_data = ['{}\t{}\n'.format(image_path, label) for image_path in sub_folder.rglob('*') if is_image_file(str(image_path))]
        random.shuffle(train_test_data)
        length = len(train_test_data)
        test_length = int(length * test_ratio)
        all_train_data.extend(train_test_data[test_length:])
        all_test_data.extend(train_test_data[:test_length])
    with open(train_txt, 'w') as f:
        f.writelines(all_train_data)
    with open(test_txt, 'w') as f:
        f.writelines(all_test_data)


if __name__ == '__main__':
    image_folder = Path('/home/work/dataset/public/cat_dog')
    train_txt = './train_cat_dog.txt'
    test_txt = './test_cat_dog.txt'
    make_train_test_txt(image_folder, train_txt, test_txt, test_ratio=0.1, check_label=True)
