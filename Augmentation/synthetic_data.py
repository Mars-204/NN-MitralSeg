# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import cv2
import numpy as np
import pytorchvideo
import torch
import os
import matplotlib.pyplot as plt
import augmentations
from torchvision import transforms
from numpy import asarray
from os.path import isfile, join
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
        self.video_list = os.listdir(os.path.join(dir_path, str(data_folder)))
        self.fps = 50
        #import ipdb; ipdb.set_trace()
        for i, vid in enumerate(self.video_list):
            self.video_list[i] = os.path.join( str(data_folder),vid)


    def syn_video(self):

        for i in range(len(self.video_list)):
            # Converting video to frames and applying augmix operation on them
            frames = self.vid2frame(self.video_list[i])

            # Converting frames to video
            aug_video = self.frame2vid(frames)
            # Storing synthetic video in new folder

        
        return aug_video
            


    def vid2frame(self,video):
        """
        Converting video to frame and storing it in new folder

        Converting video to 3d matrix
        """
        cap = cv2.VideoCapture(video)
        count = 0
        try:
        # creating a folder named data
            if not os.path.exists('data'):
                os.makedirs('data')
        
        # if not created then raise error
        except OSError:
            print ('Error: Creating directory of data')

        sec = 0
        while cap.isOpened():
            sec = sec + self.fps
            sec = round(sec,2)
            cap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
            ret, frame = cap.read()
            if ret:
                # cv2.imshow('frame', frame); cv2.waitKey(10)
                frame = transforms.ToPILImage()(frame)
                frame = self.aug (frame,self.preprocess)
                frame = asarray(frame)
                #cv2.imshow('frame', frame); cv2.waitKey(10) 
                name = './data/frame' + str(count) + '.jpg'
                cv2.imwrite(name, frame)
                count += 1
            else:
                break
        cap.release()
        cv2.destroyAllWindows()

    def frame2vid(self, frames):
        """
        Converting frames to video and storing it in new folder

        """
        pathIn= '/home/patel/mitral/NN-MitralSeg/Augmentation/data/'
        pathOut = '/home/patel/mitral/NN-MitralSeg/Augmentation/Echo_data/EchoNet-Dynamic/Aug_Videos/video.avi'

        frame_array = []
        files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
        #for sorting the file names properly
        files.sort(key = lambda x: x[5:-4])
        files.sort()
        frame_array = []
        files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
        #for sorting the file names properly
        files.sort(key = lambda x: x[5:-4])
        #import ipdb; ipdb.set_trace()
        for i in range(len(files)):
            filename=pathIn + files[i]
            #reading each files
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width,height)
            
            #inserting the frames into an image array
            frame_array.append(img)
        out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), self.fps, size)
        for i in range(len(frame_array)):
            # writing to a image array
            out.write(frame_array[i])
        out.release()



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
                op = aug_list[1]
                image_aug = op(image_aug, aug_severity)
            # Preprocessing commutes since all coefficients are convex
            mix += ws[i] * self.preprocess(image_aug)

        mixed = (1 - m) * self.preprocess(image) + m * mix
        mixed = transforms.ToPILImage()(mixed)
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