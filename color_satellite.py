import os
import cv2
import numpy as np

Building = '#3C1098'  # rgb 60, 16, 152
Land = '#8429F6'  # rgb 132, 41, 246
Road = '#6EC1E4'  # rgb 110, 193, 228
Vegetation = 'FEDD3A'  # rgb 254, 221, 58
Water = 'E2A929'  # rgb 226, 169, 41
Unlabeled = '#9B9B9B'  # background  # rgb 155, 155, 155

'''
for green channel
'''

a = int('10', 16)
b = int('29', 16)
c = int('C1', 16)
d = int('DD', 16)
e = int('A9', 16)
f = int('9B', 16)
print({'building': a, 'land': b, 'road': c, 'vegetation': d, 'water': e, 'background': f})

root_directory = 'sattelite_data/'

for path, _, files in os.walk(root_directory):
    dir = path.split(os.path.sep)[-1]
    if dir == 'masks':  # Find all 'masks' directories
        mask_lst = os.listdir(path)  # List of all masks names in this subdirectory
        for msk in mask_lst:
            mask = cv2.imread(os.path.join(path, msk))
            mask_g = mask[:, :, 1].copy()
            mask_g[mask_g == 41] = 0  # land
            mask_g[mask_g == 16] = 1  # building
            mask_g[mask_g == 193] = 2  # road
            mask_g[mask_g == 221] = 3  # vegetation
            mask_g[mask_g == 169] = 4  # water
            mask_g[mask_g == 155] = 5  # background
            print(np.unique(mask_g))
