from keras import models # load model
from PIL import Image # open and resize image
import numpy as np 

CATEGORIES = ['Metal', 'Glass', 'Biological', 'Paper', 'Battery', 'Trash', 'Cardboard', 'Shoes', 'Clothes', 'Plastic']

def predict(image_object, model):

    # resize image
    img = image_object.resize((100, 100))

    # convert image to numpy array
    img_array = np.array(img)

    # if the image is grayscale, convert it to RGB
    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)

    # ensure the image has 3 color channels
    if img_array.shape[2] == 4:
        img_array = img_array[..., :3]

    # reshape to required shape
    img_array = img_array.reshape(1, 100, 100, 3)

    # convert to float32
    img_array = img_array.astype('float32')
    
    # max-min normalize
    img_array /= 255
    
    output = model.predict(x=img_array, batch_size=1, verbose=0)
    return CATEGORIES[np.argmax(output)]

model = models.load_model(r'D:\งาน\KMITL\ปี3\ปี3เทอม2\Introduction to Data Analytics\Assignment\Project\Source\best_model.h5')

img = Image.open(r'D:\งาน\KMITL\ปี3\ปี3เทอม2\Introduction to Data Analytics\Assignment\Project\Garbage Classification\plastic\plastic_0037.jpg')
output = predict(img, model)
print(output)