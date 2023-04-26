import cv2
import os
import numpy as np
import pickle

ROOT_PATH = 'Garbage Classification'
CATEGORIES = ['metal', 'glass', 'biological', 'paper', 'battery', 'trash', 'cardboard', 'shoes', 'clothes', 'plastic']
IMG_SIZE = (100, 100)

img_dict = {'file_name': [],
            'image': [],
            'class_name': CATEGORIES,
            'class_no': []
            }

for cat in CATEGORIES:
    path = os.path.join(ROOT_PATH, cat)
    class_num = CATEGORIES.index(cat)
    labels = np.zeros(shape=(len(CATEGORIES)))
    labels[class_num] = 1
    print('start loading: ' + cat)
    number = 0
    for img_path in os.listdir(path):
        img = cv2.imread(os.path.join(path, img_path))              # read image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)                  # convert image to RGB ordering (defalut->BGR)
        resized_img = cv2.resize(img, IMG_SIZE)                     # resize image
        img_dict['file_name'].append(img_path)
        img_dict['image'].append(resized_img)
        img_dict['class_no'].append(class_num)
        number += 1                                                 # count image
    print('finish loading: ' + str(number) + ' image\n')

img_dict['image'] = np.array(img_dict['image'], dtype='float32')

print(img_dict['file_name'][:5], end='\n\n')
print(img_dict['image'][:5], end='\n\n')
print(img_dict['class_name'][:5], end='\n\n')
print(img_dict['class_no'][:5], end='\n\n')

with open(r'Garbage Classification\img_dict.obj', 'wb') as file:    # save dict
    pickle.dump(img_dict, file)