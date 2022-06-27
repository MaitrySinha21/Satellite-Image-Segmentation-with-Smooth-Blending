from keras.models import load_model
import os
import random
import cv2
import numpy as np
from matplotlib import pyplot as plt
root_dir = 'satellite_data'


model = load_model('Flask/model/satellite_256_unet_model_c6.hdf5', compile=False)
test_img_dir = "{}/data/training_data/train_images/train/".format(root_dir)
test_mask_dir = "{}/data/training_data/train_masks/train/".format(root_dir)
imgLst = os.listdir(test_img_dir)
mskLst = os.listdir(test_mask_dir)
img_number = random.randint(0, len(imgLst))


def random_prediction():
    ground_truth = cv2.imread(test_mask_dir + '{}'.format(mskLst[img_number]), 0)
    image = cv2.imread(test_img_dir + '{}'.format(imgLst[img_number]))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    test_img = (image.reshape(-1, image.shape[-1])).reshape(image.shape)/255.0
    test_img_input = np.expand_dims(test_img, 0)
    prediction = model.predict(test_img_input)
    predicted_img = np.argmax(prediction, axis=3)[0, :, :]
    plt.figure(figsize=(12, 6))

    plt.subplot(231)
    plt.imshow(test_img)
    plt.title('Test Image')
    plt.axis('off')

    plt.subplot(232)
    plt.imshow(ground_truth)
    plt.title('Original Label')
    plt.axis('off')

    plt.subplot(233)
    plt.imshow(predicted_img)
    plt.title('Predicted Mask')
    plt.axis('off')
    plt.show()


# random_prediction()
