"""
code : swati sinha & maitry sinha
"""
from cat_generator import *
from data import find_labels_after_patches, data_size, batch
import tensorflow as tf
from unet_model import unet_model
# from unet_model import jacard_coef, jacard_coef_loss, dice_coef, dice_coef_loss
import os

size = data_size()
batch_size = batch()
n_class, labels = find_labels_after_patches()

root_dir = 'satellite_data'
model_path = 'Flask/model/satellite_256_unet_model_c6.hdf5'


class Training:
    def __init__(self, n_class, model_path, root_dir, batch_size, size):
        self.batch_size = batch_size
        self.n_class = n_class
        self.model_path = model_path
        self.directory = root_dir
        self.size = size
        self.train_img_path = "{}/data/training_data/train_images/".format(self.directory)
        self.train_mask_path = "{}/data/training_data/train_masks/".format(self.directory)
        self.val_img_path = "{}/data/training_data/val_images/".format(self.directory)
        self.val_mask_path = "{}/data/training_data/val_masks/".format(self.directory)
        self.train_img_gen = trainGenerator(self.train_img_path, self.train_mask_path,
                                            batch_size=self.batch_size, img_size=self.size, n_class=self.n_class)
        self.val_img_gen = trainGenerator(self.val_img_path, self.val_mask_path,
                                          batch_size=self.batch_size, img_size=self.size, n_class=self.n_class)
        self.num_train_imgs = len(os.listdir(os.path.join(self.train_img_path, 'train')))
        self.num_val_imgs = len(os.listdir('{}/data/training_data/val_images/val/'.format(self.directory)))
        self.steps_per_epoch = self.num_train_imgs // self.batch_size
        self.val_steps_per_epoch = self.num_val_imgs // self.batch_size

    def print_gen(self):
        print('No. of train images     :', self.num_train_imgs)
        print('No. of validation images:', self.num_val_imgs)
        print('training steps per epoch:', self.steps_per_epoch)
        print('validation steps per epoch:', self.val_steps_per_epoch)

    def model_save(self):
        mod = unet_model(size=(self.size, self.size, 3), n_class=self.n_class)
        os.makedirs('unet_models', exist_ok=True)
        mod.save(self.model_path)
        print('model saved at....:{}'.format(self.model_path))

    def train(self, epochs=10, lr=2e-4, training_type='saved_model'):
        if training_type == 'saved_model':
            model = tf.keras.models.load_model(self.model_path, compile=True)
            print('saved model loaded..')
        elif training_type == 'new_model':
            self.model_save()
            model = tf.keras.models.load_model(self.model_path, compile=True)
            print('new model loaded...')

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                      loss='categorical_crossentropy', metrics='accuracy')
        hist = model.fit(self.train_img_gen, steps_per_epoch=self.steps_per_epoch,
                         epochs=epochs,
                         verbose=1, validation_data=self.val_img_gen,
                         validation_steps=self.val_steps_per_epoch)
        model.save(self.model_path)
        print('model trained and saved..')
        return hist


train = Training(model_path=model_path, root_dir=root_dir, n_class=n_class, batch_size=batch_size, size=size)
# train.print_gen()
"""
for new data/model write 'new_model' 
"""
hist = train.train(epochs=1, lr=1e-7, training_type='saved_model')
# plot_history(hist)
