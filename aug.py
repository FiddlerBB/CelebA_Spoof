import albumentations as A
import cv2
import tensorflow as tf

AUGMENTATION_TRAIN = A.Compose([
    A.HorizontalFlip(),
    A.Rotate(limit=30),
    A.ChannelShuffle(),
    A.PadIfNeeded(min_height=128, min_width=128, p=1),
    A.CenterCrop(224, 224, p=0.5),
    A.RandomCrop(width=224, height=224, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5,
                       border_mode=cv2.BORDER_REFLECT),
    A.ColorJitter(brightness=0.07, contrast=0.07, saturation=0.1, hue=0.1, always_apply=False, p=0.3),
    A.RGBShift()
])