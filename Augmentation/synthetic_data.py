# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import cv2
import numpy as np
import pytorchvideo
import torch
import os
import matplotlib.pyplot as plt
import augmentations
from torchvision import transforms
#import tensorflow as tf
from pytorchvideo.transforms import AugMix 



from pytorchvideo.transforms.augmentations import (
    _AUGMENTATION_MAX_LEVEL,
    AugmentTransform,
    _decreasing_int_to_arg,
    _decreasing_to_arg,
    _increasing_magnitude_to_arg,
    _increasing_randomly_negate_to_arg,
)
from pytorchvideo.transforms.transforms import OpSampler


_AUGMIX_LEVEL_TO_ARG = {
    "AutoContrast": None,
    "Equalize": None,
    "Rotate": _increasing_randomly_negate_to_arg,
    "Posterize": _decreasing_int_to_arg,
    "Solarize": _decreasing_to_arg,
    "ShearX": _increasing_randomly_negate_to_arg,
    "ShearY": _increasing_randomly_negate_to_arg,
    "TranslateX": _increasing_randomly_negate_to_arg,
    "TranslateY": _increasing_randomly_negate_to_arg,
    "AdjustSaturation": _increasing_magnitude_to_arg,
    "AdjustContrast": _increasing_magnitude_to_arg,
    "AdjustBrightness": _increasing_magnitude_to_arg,
    "AdjustSharpness": _increasing_magnitude_to_arg,
}

_TRANSFORM_AUGMIX_MAX_PARAMS = {
    "AutoContrast": None,
    "Equalize": None,
    "Rotate": (0, 30),
    "Posterize": (4, 4),
    "Solarize": (1, 1),
    "ShearX": (0, 0.3),
    "ShearY": (0, 0.3),
    "TranslateX": (0, 1.0 / 3.0),
    "TranslateY": (0, 1.0 / 3.0),
    "AdjustSaturation": (0.1, 1.8),
    "AdjustContrast": (0.1, 1.8),
    "AdjustBrightness": (0.1, 1.8),
    "AdjustSharpness": (0.1, 1.8),
}

# Hyperparameters for sampling magnitude.
# sampling_data_type determines whether uniform sampling samples among ints or floats.
# sampling_min determines the minimum possible value obtained from uniform
# sampling among floats.
SAMPLING_AUGMIX_DEFAULT_HPARAS = {"sampling_data_type": "float", "sampling_min": 0.1}


data_folder = "/home/patel/mitral/NN-MitralSeg/Augmentation/Echo_data/EchoNet-Dynamic/Videos"
dir_path = "/home/patel/mitral/NN-MitralSeg/Augmentation/Echo_data/EchoNet-Dynamic"

class syn_data(object):

    def __init__(self, data_folder, dir_path):

        self.data_folder = data_folder
        self.dir_path = dir_path
        # Preprocess for the AugMix function
        self.preprocess = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)])

        # Loading videos from the data folder
        self.video_list = os.listdir(os.path.join(self.dir_path, str(self.data_folder)))


    def syn_video(self):

        for i in range(len(self.video_list)):
            # Converting video to frames
            frames = self.vid2frame(self.video_list[i])
            # Applying AugMix operation on video data and storing in new folder
            syn_frames = self.aug(frames)
            # Converting frames to video
            aug_video = self.frame2vid(syn_frames)
            # Storing synthetic video in new folder

        
        return aug_video
            


    def vid2frame(self,video):
        """
        Converting videos to frame and storing t in new folder
        """
        video = cv2.VideoCapture(video)
        f = os.path(self.dir_path + "/frames")
        while(video.isOpened()):
            ret, frame = video.read()
            print(ret)
            if ret == False:
                break
            cv2.imshow('frame', frame); cv2.waitKey(0)
            f = cv2.imwrite('my_video_frame.png', frame)

        video.release()
        cv2.destroyAllWindows()
        return f

    def frame2vid(self, frames):
        """
        Converting frames to video and storing it in new folder

        """



    """
    Implementation of AUGMIX as implemented in AUGMIX paper

    """

    CORRUPTIONS = [
        'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
        'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
        'brightness', 'contrast', 'elastic_transform', 'pixelate',
        'jpeg_compression'
    ]


    def aug(self,image, preprocess):
        """Perform AugMix augmentations and compute mixture.

        Args:
            image: PIL.Image input image
            preprocess: Preprocessing function which should return a torch tensor.

        Returns:
            mixed: Augmented and mixed image.
        """
        try:
                all_ops = config.all_ops
                mixture_width = config.mixture_width
                mixture_depth = config.mixture_depth
                aug_severity = config.aug_severity
        except Exception as e:
                all_ops = True
                mixture_depth = -1
                mixture_width = 3
                aug_severity = 3
            
            
        aug_list = augmentations.augmentations_all

        ws = np.float32(np.random.dirichlet([1] * mixture_width))
        m = np.float32(np.random.beta(1, 1))

        mix = torch.zeros_like(self.preprocess(image))
        for i in range(mixture_width):
            image_aug = image.copy()
            depth = mixture_depth if mixture_depth > 0 else np.random.randint(
                1, 4)
            for _ in range(depth):
                op = np.random.choice(aug_list)
                image_aug = op(image_aug, aug_severity)
            # Preprocessing commutes since all coefficients are convex
            mix += ws[i] * self.preprocess(image_aug)

        mixed = (1 - m) * self.preprocess(image) + m * mix
        return mixed


# class AugMixDataset(torch.utils.data.Dataset):
#   """Dataset wrapper to perform AugMix augmentation."""

#   def __init__(self, dataset):
#     self.dataset = dataset

#   def __getitem__(self, i):
#     x, y = self.dataset[i]
#     im_tuple = (x, aug(x),aug(x))
#     return im_tuple, y

#   def __len__(self):
#     return len(self.dataset)