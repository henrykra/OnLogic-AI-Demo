import pandas as pd
import json
import re
import os

df = pd.read_csv('csv_labels_new.csv')

labels_dict = {
    "Captain Antilles": 0,
    "Imperial Stormtrooper": 1,
    "Rebel Trooper": 2,
    "Clone Trooper": 3
}

labels_path = 'labels/'

if not os.path.exists(labels_path):
    os.makedirs(labels_path)


for row in df.iterrows():
    labels = row[1]['label']
    
    img_fname = row[1]['image']

    img = re.search(r'img\d+', img_fname)[0]

    json_labels = json.loads(labels)



    with open(labels_path + img + '.txt', 'w') as txtfile:

        for label in json_labels:
            new_x = label['x'] / 100 + label['width'] / 200
            new_y = label['y'] / 100 + label['height'] / 200
            new_width = label['width'] / 100
            new_height = label['height'] / 100

            txtfile.write(str(labels_dict[label['rectanglelabels'][0]]) + '\t')
            txtfile.write(str(new_x) + '\t')
            txtfile.write(str(new_y) + '\t')
            txtfile.write(str(new_width) + '\t')
            txtfile.write(str(new_height) + '\n')    
    
