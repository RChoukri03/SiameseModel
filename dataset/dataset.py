import os
import random
from time import time
import traceback
import torch
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import cv2
import albumentations as A
from sklearn.preprocessing import LabelEncoder

# Configuration de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ImagePairsDataset(Dataset):
    def __init__(self, img_labels, img_dir, augmentations=None):
        self.img_labels = img_labels
        self.img_dir = img_dir
        random.seed(int(time()))
        self.augmentations1 = augmentations or A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.2, p=0.35),
            A.RandomGamma(gamma_limit=(80, 120), p=0.35),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=8, val_shift_limit=4, p=0.35),
            A.ShiftScaleRotate(shift_limit=0.00001, scale_limit=0.05, rotate_limit=2, border_mode=cv2.BORDER_REFLECT_101, p=0.35, interpolation=3),
            A.ColorJitter(p=0.4),
            A.Perspective (p=0.6),
            A.OneOf([
            A.OpticalDistortion(p=0.4),
            A.GridDistortion(p=.2),
            A.IAAPiecewiseAffine(p=0.4),
            ], p=0.3),
            A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            )
        ])
        random.seed(int(time()))
        self.augmentations2 = augmentations or A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.2, p=0.35),
            A.RandomGamma(gamma_limit=(80, 120), p=0.35),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=8, val_shift_limit=4, p=0.35),
            A.ShiftScaleRotate(shift_limit=0.00001, scale_limit=0.05, rotate_limit=2, border_mode=cv2.BORDER_REFLECT_101, p=0.35, interpolation=3),
            A.ColorJitter(p=0.4),
            A.Perspective (p=0.6),
            A.OneOf([
            A.OpticalDistortion(p=0.4),
            A.GridDistortion(p=.2),
            A.IAAPiecewiseAffine(p=0.4),
            ], p=0.3),
            A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            )
        ])
        self.label_encoder = LabelEncoder()
        try:
            self.img_labels['label'] = self.label_encoder.fit_transform(self.img_labels['label'])
        except Exception as e:
            logging.error(f"Error in label encoding: {e}")
            raise

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        try:
            img_path1 = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
            img_path2 = os.path.join(self.img_dir, self.img_labels.iloc[idx, 1])
            image1, image2 = cv2.imread(img_path1), cv2.imread(img_path2)
            image1 = cv2.resize(image1, (224,224))
            image2 = cv2.resize(image2, (224,224))
            if image1 is None or image2 is None:
                raise ValueError(f"Image not found at {img_path1} or {img_path2}")

            image1, image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB), cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

            if self.augmentations1 and self.augmentations2:
                augmented1 = self.augmentations1(image=image1)
                image1 = augmented1['image']
                augmented2 = self.augmentations2(image=image2)
                image2 = augmented2['image']

            # Convertir les images en tensors PyTorch
            image1 = torch.from_numpy(image1.transpose(2, 0, 1)).type(torch.FloatTensor)
            image2 = torch.from_numpy(image2.transpose(2, 0, 1)).type(torch.FloatTensor)

            label = torch.tensor(self.img_labels.iloc[idx, 2], dtype=torch.float32)
            return (image1, image2), label
        except Exception as e:
            logging.error(f"Error loading image pair at index {idx}: {e}")
            traceback.print_exc()
            raise

def createDatasets(csvFile, imgDir, splitRatio=0.8, augmentations=None):
    try:
        logging.info(f"Reading CSV file: {csvFile}")
        data = pd.read_csv(csvFile)
        trainData, valData = train_test_split(data, test_size=1-splitRatio, shuffle=True)
        trainDataset = ImagePairsDataset(trainData, imgDir)
        valDataset = ImagePairsDataset(valData, imgDir)
        return trainDataset, valDataset
    except Exception as e:
        logging.error(f"Error creating datasets: {e}")
        raise