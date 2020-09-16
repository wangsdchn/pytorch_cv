import json
from collections import OrderedDict
from pathlib import Path
import os


def read_coco():
    json_root = Path('/home/work/dataset/public/coco/annotations/')
    for item in ['val', 'train']:
        instances_json_file = json_root / 'instances_{}2017.json'.format(item)
        with open(instances_json_file, 'r') as f:
            for line in f:
                json_data = json.loads(line)
        data = []
        for image_info in json_data['images']:
            data.append('{}\n'.format(json.dumps(image_info)))
        with open('{}_images.json'.format(item), 'w') as f:
            f.writelines(data)
        data = []
        for image_info in json_data['annotations']:
            data.append('{}\n'.format(json.dumps(image_info)))
        with open('{}_annotations.json'.format(item), 'w') as f:
            f.writelines(data)
        data = []
        for image_info in json_data['categories']:
            data.append('{}\n'.format(json.dumps(image_info)))
        with open('{}_categories.json'.format(item), 'w') as f:
            f.writelines(data)


def make_dataset():
    categories_dict, categories_dict_new = {}, {}
    categories_list_new = []
    with open('train_categories.json', 'r') as f:
        for line in f:
            data = json.loads(line.strip('\n'))
            categories_dict[data['id']] = data['name']
    for idx, (key, data) in enumerate(sorted(categories_dict.items())):
        new_data = {}
        new_data['label'] = idx
        new_data['name'] = data
        categories_list_new.append('{}\n'.format(json.dumps(new_data)))
        categories_dict_new[data] = idx

    for item in ['train', 'val']:
        all_data = []
        categories_dict = {}
        annotations_dict = {}
        categories_file = '{}_categories.json'.format(item)
        images_file = '{}_images.json'.format(item)
        annotations_file = '{}_annotations.json'.format(item)
        with open(categories_file, 'r') as f:
            for line in f:
                data = json.loads(line.strip('\n'))
                categories_dict[data['id']] = data['name']
        with open(annotations_file, 'r') as f:
            for line in f:
                data = json.loads(line.strip('\n'))
                image_id = data['image_id']
                del data['segmentation']
                del data['image_id']
                del data['id']
                data['name'] = categories_dict[data['category_id']]
                del data['category_id']
                data['label'] = categories_dict_new[data['name']]
                if image_id in annotations_dict:
                    annotations_dict[image_id].append(data)
                else:
                    annotations_dict[image_id] = [data]
        with open(images_file, 'r') as f:
            for line in f:
                data = json.loads(line.strip('\n'))
                image_id = data['id']
                image_name = data['file_name']
                new_dict = OrderedDict()
                new_dict[image_name] = image_name
                if image_id in annotations_dict:
                    new_dict['info'] = annotations_dict[image_id]
                all_data.append('{}\n'.format(json.dumps(new_dict)))
        os.remove(annotations_file)
        # os.remove(categories_file)
        os.remove(images_file)

        with open('coco_{}.json'.format(item), 'w') as f:
            f.writelines(all_data)
    with open('coco_category.json', 'w') as f:
        f.writelines(categories_list_new)


if __name__ == '__main__':
    read_coco()
    make_dataset()
