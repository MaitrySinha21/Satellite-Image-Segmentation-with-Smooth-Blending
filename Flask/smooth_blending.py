"""
ref:- https://github.com/Vooban/Smoothly-Blend-Image-Patches
"""
import numpy as np
from tiled_predictions import smooth_windowing
from matplotlib import pyplot as plt


def satellite_mask_color_restoration(mask):
    mask = np.array(mask, dtype='uint8')
    mask = np.dstack((mask, mask, mask))
    mask[:, :, 0][mask[:, :, 0] == 1] = 63
    mask[:, :, 0][mask[:, :, 0] == 2] = 117
    mask[:, :, 0][mask[:, :, 0] == 3] = 110
    mask[:, :, 0][mask[:, :, 0] == 4] = 18
    mask[:, :, 0][mask[:, :, 0] == 5] = 226
    mask[:, :, 0][mask[:, :, 0] == 0] = 155

    mask[:, :, 1][mask[:, :, 1] == 1] = 60
    mask[:, :, 1][mask[:, :, 1] == 2] = 17
    mask[:, :, 1][mask[:, :, 1] == 3] = 193
    mask[:, :, 1][mask[:, :, 1] == 4] = 128
    mask[:, :, 1][mask[:, :, 1] == 5] = 169
    mask[:, :, 1][mask[:, :, 1] == 0] = 155

    mask[:, :, 2][mask[:, :, 2] == 1] = 232
    mask[:, :, 2][mask[:, :, 2] == 2] = 8
    mask[:, :, 2][mask[:, :, 2] == 3] = 228
    mask[:, :, 2][mask[:, :, 2] == 4] = 40
    mask[:, :, 2][mask[:, :, 2] == 5] = 41
    mask[:, :, 2][mask[:, :, 2] == 0] = 155

    return mask


def Final_prediction(img, model, patch_size, n_classes):
    img = img[:, :, ::-1]/255.0
    input_img = (img.reshape(-1, img.shape[-1])).reshape(img.shape)
    predictions_smooth = smooth_windowing(
        input_img,
        window_size=patch_size,
        subdivisions=2,
        nb_classes=n_classes,
        pred_func=(
            lambda img_batch_subdiv: model.predict((img_batch_subdiv))
        )
    )

    final_prediction = np.argmax(predictions_smooth, axis=2)
    col_mask = satellite_mask_color_restoration(final_prediction)
    plt.imsave('static/mask.png', col_mask)
    plt.imsave('static/input.jpg', img)
