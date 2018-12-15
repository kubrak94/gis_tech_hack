import os
from torch.utils.data import Dataset

import cv2
import numpy as np

import utils.transforms as transforms


class ImageMaskCSVDataset(Dataset):
    def __init__(self, labels, 
                 image_column, mask_column, 
                 images_folder, masks_folder, 
                 transform=None):
        
        self.transform = transform
        self.labels = labels

        self.image_column = image_column
        self.mask_column = mask_column

        self.images_folder = images_folder
        self.masks_folder = masks_folder

    def process_new_item(self, index):
        row = self.labels.iloc[index]

        file_path = row[self.image_column]
        image = cv2.imread(os.path.join(self.images_folder, file_path))[:, :, ::-1]

        mask_path = row[self.mask_column]
        mask = cv2.imread(os.path.join(self.masks_folder, mask_path))[:, :, 0]

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        sample = {'image': image,
                  'mask': mask
                 }

        return sample

    def __getitem__(self, index):
        result = self.process_new_item(index)
        return result

    def __len__(self):
        return len(self.labels)


class RandomCropDataset(Dataset):
    def __init__(self, labels, 
                 image_column, mask_column, 
                 images_folder, masks_folder,
                 h_size, w_size, rnd_crop, crop_range,
                 transform=None):
        
        self.transform = transform
        self.labels = labels
        self.image_column = image_column
        self.mask_column = mask_column
        self.images_folder = images_folder
        self.masks_folder = masks_folder
        self.h_size = h_size
        self.w_size = w_size
        self.rnd_crop = rnd_crop
        self.crop_range = crop_range
        
        self.images = []
        self.masks = []
        self.count = 0
        
        labels = labels
        for row in labels.iterrows():
            img_path = row[1][image_column]
            mask_path = row[1][mask_column]
            
            img = cv2.imread(img_path)[..., ::-1]
            mask = cv2.imread(mask_path)[..., 0]
            
            self.images.append(img)
            self.masks.append(mask)
            self.count += 1
        
        self.n_rows = mask.shape[0] // h_size
        self.n_cols = mask.shape[1] // w_size

    def process_new_item(self, index):
        img_idx = index // (self.n_rows * self.n_cols)
        pidx = index - img_idx * (self.n_rows * self.n_cols)
        
        img = self.images[img_idx]
        mask = self.masks[img_idx]
        
        if not self.rnd_crop:
            row_idx = pidx // self.n_cols
            col_idx = pidx % self.n_cols
        
            img_p = img[row_idx * self.h_size: row_idx * self.h_size + self.h_size,
                        col_idx * self.w_size: col_idx * self.w_size + self.w_size]
            mask_p = mask[row_idx * self.h_size: row_idx * self.h_size + self.h_size,
                          col_idx * self.w_size: col_idx * self.w_size + self.w_size]
            
#             d_mask = np.zeros_like(mask)
#             d_mask[row_idx * self.h_size: row_idx * self.h_size + self.h_size,
#                    col_idx * self.w_size: col_idx * self.w_size + self.w_size] = 255
#             imshow(d_mask)
        else:
            size = np.random.randint(self.crop_range[0], self.crop_range[1])
            x = np.random.randint(0, self.n_rows * self.h_size - size)
            y = np.random.randint(0, self.n_cols * self.w_size - size)
            
            img_p = img[x:x + size, y: y + size]
            mask_p = mask[x:x + size, y: y + size]
            
#             d_mask = np.zeros_like(mask)
#             d_mask[x:x + size, y: y + size] = 255
#             imshow(d_mask)
        
        if self.transform is not None:
            augmented = self.transform(image=img_p, mask=mask_p)
            img_p = augmented['image']
            mask_p = augmented['mask']

        sample = {'image': img_p,
                  'mask': mask_p
                 }

        return sample

    def __getitem__(self, index):
        result = self.process_new_item(index)
        return result

    def __len__(self):
        return self.n_rows * self.n_cols * self.count
