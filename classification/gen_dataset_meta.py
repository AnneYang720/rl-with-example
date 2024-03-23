from classes import IMAGENET2012_CLASSES
import csv
import os

# load imagenet-1k test set from local dir
data_path = '/home/gyc/datasets/imagenet-tiny/val'

all_class_to_num = {class_name: i for i, class_name in enumerate(IMAGENET2012_CLASSES)}

with open(data_path + '/metadata.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['file_name', 'label', 'class_idx', 'class_name'])
    for file_name in sorted(os.listdir(data_path)):
        if not file_name.endswith('.JPEG'):
            continue

        class_idx = file_name.split('_')[3].split('.')[0]
        class_name = IMAGENET2012_CLASSES[class_idx]
        class_number = all_class_to_num[class_idx]
        writer.writerow([file_name, class_number, class_idx, class_name])