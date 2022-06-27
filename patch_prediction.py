from keras.models import load_model
import os
import random
import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
root_directory = 'nucleus_data'
model = load_model('Flask_nucleus/model/nucleus_small_256_unet_model.hdf5', compile=False)

test_img_dir = "{}/patches/images/".format(root_directory)
imgLst = os.listdir(test_img_dir)
img_number = random.randint(0, len(imgLst))


def random_prediction(model):
    image = cv2.imread(test_img_dir+f'{imgLst[img_number]}')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    test_img = (image.reshape(-1, image.shape[-1])).reshape(image.shape)/255.0
    test_img_input=np.expand_dims(test_img, 0)
    prediction = model.predict(test_img_input)
    predicted_img = np.squeeze(prediction)
    predicted_img = np.round(predicted_img).astype('int')
    plt.figure(figsize=(20, 20))
    plt.subplot(221)
    plt.title('Testing Image')
    plt.imshow(test_img)
    plt.axis('off')
    plt.subplot(222)
    plt.title('Test prediction')
    plt.imshow(predicted_img)
    plt.axis('off')
    plt.show()


random_prediction(model)