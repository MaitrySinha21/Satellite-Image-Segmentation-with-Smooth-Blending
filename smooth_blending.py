"""
ref:- https://github.com/Vooban/Smoothly-Blend-Image-Patches
"""
import numpy as np
import cv2
from tiled_predictions import smooth_windowing
from matplotlib import pyplot as plt
from skimage import color


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


def neucleus_color_mask(mask, col=(155, 40, 52)):
    mask[:, :, 0][mask[:, :, 0] == 1] = col[0]
    mask[:, :, 1][mask[:, :, 1] == 1] = col[1]
    mask[:, :, 2][mask[:, :, 2] == 1] = col[2]
    return  mask


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
    plt.imsave('output/out.png', col_mask)
    plt.imsave('output/out.jpg', img)
    plt.imshow(col_mask)
    plt.show()


class Binary_mask_instance_segmentation:

    def __init__(self, mask, border_size=5):
        self.mask = mask * 255
        self.border_size = border_size
        self.actual_image = np.dstack((self.mask, self.mask, self.mask))

    def generate_border(self):
        erosion_kernel = np.ones((3, 3), np.uint8)
        erode_img = cv2.erode(self.mask, erosion_kernel, iterations=1)
        kernel_size = 2 * self.border_size + 1

        dialation_kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilated = cv2.dilate(erode_img, dialation_kernel, iterations=1)
        dilated127 = np.where(dilated == 255, 127, dilated)
        with_border = np.where(erode_img > 127, 255, dilated127)

        return with_border

    def segmentation(self):
        mask = self.generate_border()
        mask[mask == 127] = 1
        mask[mask == 255] = 2

        sure_fg = (mask == 2).astype(np.uint8)
        sure_fg[sure_fg == 1] = 2
        unknown = (mask == 1).astype(np.uint8)

        ret, markers = cv2.connectedComponents(sure_fg)
        numbers = len(np.unique(markers)) - 2
        markers = markers + 10
        markers[unknown == 1] = 0

        markers = cv2.watershed(self.actual_image, markers)
        markers[markers == -1] = 0
        segmented_img = color.label2rgb(markers, bg_label=0)

        return numbers, segmented_img * 255


def Final_binary_prediction(img, model, patch_size, col=(155, 40, 52)):
    img = np.expand_dims(img, axis=2)
    input_img = img/255.0
    predictions_smooth = smooth_windowing(
        input_img,
        window_size=patch_size,
        subdivisions=2,
        nb_classes=1,
        pred_func=(
            lambda img_batch_subdiv: model.predict((img_batch_subdiv))
        )
    )
    predictions_smooth = np.squeeze(predictions_smooth)
    final_prediction = np.round(predictions_smooth).astype('int')
    mask = np.dstack((final_prediction, final_prediction, final_prediction))
    mask = neucleus_color_mask(mask, col=(235, 52, 52))
    img = np.squeeze(img, axis=2)
    cv2.imwrite('output/binary_col_mask.png', mask[:, :, ::-1])
    cv2.imwrite('output/binary_image.png', img)
    msk = cv2.imread('output/binary_image.png', 0)
    s = Binary_mask_instance_segmentation(mask=msk)
    number, segment = s.segmentation()
    cv2.imwrite('output/segment.png', segment)
    print(number)
    plt.imshow(mask)
    plt.show()
    plt.imshow(segment[:, :, ::-1])
    plt.axis('off')
    plt.show()



