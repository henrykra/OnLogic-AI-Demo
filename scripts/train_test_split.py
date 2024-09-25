import os
import random

# file variables
data_dir = 'datasets/lego'
images_dir = os.path.join(data_dir, 'images')
labels_dir = os.path.join(data_dir, 'labels')
splits = ['train', 'val']
train_size = .8

# creating train and val forlders
for split in splits:
    if not os.path.exists(os.path.join(images_dir, split)):
        os.makedirs(os.path.join(images_dir, split))

    if not os.path.exists(os.path.join(labels_dir, split)):
        os.makedirs(os.path.join(labels_dir, split))

# splitting images and labels into train or val
for file in os.listdir(images_dir):
    if file not in splits:
        r = random.random()
        labelfile = file[:-3] + 'txt'
        if r < train_size:
            os.rename(os.path.join(images_dir, file), os.path.join(images_dir, 'train', file))
            os.rename(os.path.join(labels_dir, file), os.path.join(labels_dir, 'train', file))
        else:
            os.rename(os.path.join(images_dir, file), os.path.join(images_dir, 'val', file))
            os.rename(os.path.join(labels_dir, file), os.path.join(labels_dir, 'val', file))

# writing data.yaml
with open(os.path.join(data_dir, 'lego.yaml'), 'w') as f:
    f.write(f'path: ../{data_dir}\n')
    f.write('train: images/train\n')
    f.write('val: images/val\n')
    f.write('\n\n')
    f.write('names:\n')
    with open(os.path.join(data_dir, 'classes.txt'), 'r') as rf:
        for i, row in enumerate(rf):
            f.write(f'  {i}: {row}')

# unsure if classes.txt and notes.json need to be removed yet