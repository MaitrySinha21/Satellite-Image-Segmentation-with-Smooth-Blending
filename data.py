import os
import random
import cv2
import numpy as np
import splitfolders
from patchify import patchify
from PIL import Image
import matplotlib.pyplot as plt

patch_size = 256
batch_size = 4
# root_dir = 'nucleus_data'
root_dir = 'satellite_data'


def find_labels():
    labels = []
    for path, sub, files in os.walk(root_dir):
        dir = path.split(os.path.sep)[-1]
        if dir == 'masks':
            masks = os.listdir(path)
            for i, msk_num in enumerate(masks):
                msk = cv2.imread(path + "/" + msk_num, 1)
                label = np.unique(msk[:, :, 1])
                for la in label:
                    if la not in labels:
                        labels.append(la)
    labels = np.sort(labels)
    print('labels:', labels)
    print('number of classes:', len(labels))


def make_directories():
    try:
        os.makedirs('models', exist_ok=True)
        os.makedirs('output', exist_ok=True)
        os.makedirs('{}/patches/images'.format(root_dir), exist_ok=True)
        os.makedirs('{}/patches/masks'.format(root_dir), exist_ok=True)
        os.makedirs('{}/data/training_data/train_images/train'.format(root_dir), exist_ok=True)
        os.makedirs('{}/data/training_data/train_masks/train'.format(root_dir), exist_ok=True)
        os.makedirs('{}/data/training_data/val_images/val'.format(root_dir), exist_ok=True)
        os.makedirs('{}/data/training_data/val_masks/val'.format(root_dir), exist_ok=True)
        print("All Directories are created successfully..")
    except OSError as error:
        print("Directory can not be created")


def make_class_neucleus(mask):
    """
        for neucleus data taking green channel only...
        use this on patch_musk function...
    """
    mask_g = mask[:, :, 1].copy()
    mask_g[mask_g == 128] = 1

    return mask_g


def make_class_landcover(mask):
    """
        for landcover data taking green channel only...
        use this on patch_musk function...
    """
    return mask[:, :, 1].copy()


def make_class_satellite(mask):
    """
        for satellite data taking green channel only...
        use this on patch_musk function...
    """
    mask_g = mask[:, :, 1].copy()
    mask_g[mask_g == 41] = 1    # land
    mask_g[mask_g == 16] = 2    # building
    mask_g[mask_g == 193] = 3   # road
    mask_g[mask_g == 221] = 4   # vegetation
    mask_g[mask_g == 169] = 5   # water
    mask_g[mask_g == 155] = 0   # unlabeled
    return mask_g


def patch_image(patch_size):
    count = 0
    for path, sub, files in os.walk(root_dir):
        dir = path.split(os.path.sep)[-1]
        if dir == 'images':
            images = os.listdir(path)
            for k, image_name in enumerate(images):
                if image_name.endswith(".jpg") or image_name.endswith(".png"):
                    image = cv2.imread(path + "/" + image_name, 1)
                    wd = (image.shape[1] // patch_size) * patch_size
                    ht = (image.shape[0] // patch_size) * patch_size
                    image = Image.fromarray(image)
                    image = image.crop((0, 0, wd, ht))
                    image = np.array(image, dtype='uint8')
                    print("Patchifying image:", path + "/" + image_name)
                    patches_img = patchify(image, (patch_size, patch_size, 3), step=patch_size)
                    for i in range(patches_img.shape[0]):
                        for j in range(patches_img.shape[1]):
                            single_patch_img = patches_img[i, j, :, :]
                            cv2.imwrite('{}/patches/images/{}.jpg'.format(root_dir, count),
                                        single_patch_img[0])
                            count = count+1


def patch_mask(patch_size):
    count = 0
    for path, sub, files in os.walk(root_dir):
        dir = path.split(os.path.sep)[-1]
        if dir == 'masks':
            masks = os.listdir(path)
            for k, mask_name in enumerate(masks):
                if mask_name.endswith(".png") or mask_name.endswith(".jpg"):
                    mask = cv2.imread(path + "/" + mask_name, 1)
                    wd = (mask.shape[1] // patch_size) * patch_size
                    ht = (mask.shape[0] // patch_size) * patch_size
                    mask = Image.fromarray(mask)
                    mask = mask.crop((0, 0, wd, ht))
                    mask = np.array(mask, dtype='uint8')
                    print("Patchifying mask:", path + "/" + mask_name)
                    patches_mask = patchify(mask, (patch_size, patch_size, 3), step=patch_size)
                    for i in range(patches_mask.shape[0]):
                        for j in range(patches_mask.shape[1]):
                            patch_mask = patches_mask[i, j, :, :]
                            patch_mask = patch_mask[0]  # Drop extra dimension
                            patch_mask = make_class_satellite(patch_mask)
                            cv2.imwrite('{}/patches/masks/{}.png'.format(root_dir, count), patch_mask)
                            count = count+1


def find_labels_after_patches():
    path = '{}/patches/masks/'.format(root_dir)
    mask_list = os.listdir(path)
    labels = []
    for m in mask_list:
        msk = cv2.imread(os.path.join(path, m), 0)
        label = np.unique(msk)
        for l in label:
            if l not in labels:
                labels.append(l)
    labels = np.sort(labels)
    print('labels:', labels)
    print('number of classes:', len(labels))
    return len(labels), labels


class outliers:
    def __init__(self):
        pass

    def clean_outliers(self):
        os.makedirs('{}/useful/images'.format(root_dir), exist_ok=True)
        os.makedirs('{}/useful/masks'.format(root_dir), exist_ok=True)
        img_list = os.listdir('{}/patches/images/'.format(root_dir))
        msk_list = os.listdir('{}/patches/masks/'.format(root_dir))
        useless = 0

        for i in range(len(img_list)):
            img_name = img_list[i]
            mask_name = msk_list[i]
            temp_image = cv2.imread('{}/patches/images/{}'.format(root_dir, img_name), 1)
            temp_mask = cv2.imread('{}/patches/masks/{}'.format(root_dir, mask_name), 0)
            # mask = cv2.imread(f'{root_directory}patches/masks/{mask_name}')
            val, counts = np.unique(temp_mask, return_counts=True)
            # At least 5% useful area with labels that are not 0
            if (1 - (counts[0] / counts.sum())) > 0.05:
                cv2.imwrite('{}/useful/images/{}'.format(root_dir, img_name), temp_image)
                cv2.imwrite('{}/useful/masks/{}'.format(root_dir, mask_name), temp_mask)
            else:
                useless += 1

        print("Total useful images are: ", len(img_list) - useless)
        print("Total useless images are: ", useless)


def split_data():
    splitfolders.ratio(input='{}/patches/'.format(root_dir),
                       output='{}/data/training_and_testing/'.format(root_dir),
                       seed=42, ratio=(.9, .1), group_prefix=None)
    print('train data & validation data saved in "data/training_and_testing" Folder..')


def move_train_data():
    train_images = "{}/data/training_and_testing/train/images".format(root_dir)
    target_train_images = "{}/data/training_data/train_images/train".format(root_dir)
    train_masks = "{}/data/training_and_testing/train/masks".format(root_dir)
    target_train_masks = "{}/data/training_data/train_masks/train".format(root_dir)
    for img, msk in zip(os.listdir(train_images), os.listdir(train_masks)):
        image = cv2.imread(os.path.join(train_images, img), 1)
        cv2.imwrite(target_train_images + '/'+img, image)
        mask = cv2.imread(os.path.join(train_masks, msk), 0)
        cv2.imwrite(target_train_masks + '/' + msk, mask)
    print('training data moved...')


def move_validation_data():
    val_images = "{}/data/training_and_testing/val/images".format(root_dir)
    target_val_images = "{}/data/training_data/val_images/val".format(root_dir)
    val_masks = "{}/data/training_and_testing/val/masks".format(root_dir)
    target_val_masks = "{}/data/training_data/val_masks/val".format(root_dir)
    for img, msk in zip(os.listdir(val_images), os.listdir(val_masks)):
        image = cv2.imread(os.path.join(val_images, img), 1)
        cv2.imwrite(target_val_images + '/'+img, image)
        mask = cv2.imread(os.path.join(val_masks, msk), 0)
        cv2.imwrite(target_val_masks + '/' + msk, mask)
    print('validation data moved...')


def show_masks():
    for path, sub, files in os.walk(root_dir):
        dir = path.split(os.path.sep)[-1]
        if dir == 'masks':
            masks = os.listdir(path)
            for msk in masks:
                msk = cv2.imread(path + "/" + msk, 1)
                ht, wd = msk.shape[:2]
                msk = cv2.resize(msk, (720*wd//ht, 720), interpolation=cv2.INTER_NEAREST)
                cv2.imshow('masks', msk*5)
                cv2.waitKey(300)
                cv2.destroyAllWindows()


def show_images():
    for path, sub, files in os.walk(root_dir):
        dir = path.split(os.path.sep)[-1]
        if dir == 'images':
            images = os.listdir(path)
            for img in images:
                img = cv2.imread(path + "/" + img, 1)
                ht, wd = img.shape[:2]
                img = cv2.resize(img, (720*wd//ht, 720), interpolation=cv2.INTER_NEAREST)
                cv2.imshow('images', img)
                cv2.waitKey(300)
                cv2.destroyAllWindows()


def draw_patches():
    mask_path = '{}/data/training_and_testing/train/masks'.format(root_dir)
    mask_lst = os.listdir(mask_path)
    num = random.randint(0, len(mask_lst))
    img_path = '{}/data/training_and_testing/train/images'.format(root_dir)
    img_lst = os.listdir(img_path)
    mask = cv2.imread(os.path.join(mask_path, mask_lst[num]), 0)
    image = cv2.imread(os.path.join(img_path, img_lst[num]), 1)
    plt.figure(figsize=(16, 10))
    plt.subplot(121)
    plt.imshow(image)
    plt.title('Image-Patch')
    plt.subplot(122)
    plt.imshow(mask, cmap='gray')
    plt.title('Mask-Patch')
    plt.imshow(mask)
    plt.show()


def data_size():
    return patch_size


def batch():
    return batch_size


def root_directory():
    return root_dir
