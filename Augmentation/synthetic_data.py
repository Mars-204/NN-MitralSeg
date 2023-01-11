# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import cv2
import random
import numpy as np
import pytorchvideo
import torch
import os
import matplotlib.pyplot as plt
import augmentations
import tensorly as tl
import tensorflow as tf
import glob
import shutil

from pytorchvideo.transforms import AugMix 
from tqdm import tqdm
from natsort import natsorted
from numpy import asarray
from os.path import isfile, join
from torchvision import transforms

data_folder = "/home/patel/mitral/NN-MitralSeg/Augmentation/Echo_data/EchoNet-Dynamic/Videos"
dir_path = "/home/patel/mitral/NN-MitralSeg/Augmentation/Echo_data/EchoNet-Dynamic"
frames_dir = '/home/patel/mitral/NN-MitralSeg/Augmentation/data'

class syn_data(object):

    def __init__(self, data_folder, dir_path, frames_dir):

        self.data_folder = data_folder
        self.dir_path = dir_path
        self.frames_dir = frames_dir
        # Preprocess for the AugMix function
        self.preprocess = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)])

        # Loading videos from the data folder
        self.video_list = os.listdir(os.path.join(dir_path, str(data_folder)))
        self.fps = 1
        self.aug_list = augmentations.augmentations_all
        for i, vid in enumerate(self.video_list):
            self.video_list[i] = os.path.join( str(data_folder),vid)


    def syn_video(self):

        for i in range(len(self.video_list)):
            # Converting video to frames and applying augmix operation on them
            frames = self.vid2frame(self.video_list[i])
            # vid = self.augmix_vid(self.video_list[i])
            
            # Converting frames to video
            vid_no = i
            aug_video = self.images_to_video(vid_no,self.frames_dir,fps=30, image_format=".jpg", image_scale_factor=1)
            #aug_video = self.frame2vid(frames)

        
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
        frame_array = []
        sec = 0
        # import ipdb; ipdb.set_trace()
        rng = np.random.default_rng()
        op_list = rng.choice(8,3)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame_array.append(frame)
                frame = transforms.ToPILImage()(frame)
                frame = self.aug (frame,self.preprocess,op_list)
                frame = asarray(frame)
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

    def augmix_vid(self,video):
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
        frame_array = []
        sec = 0
        while cap.isOpened():
            # sec = sec + self.fps
            # sec = round(sec,2)
            # cap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
            ret, frame = cap.read()
            if ret:
                frame_array.append(frame)
                # cv2.imshow('frame', frame); cv2.waitKey(10)
                # frame = transforms.ToPILImage()(frame)
                # frame = self.aug (frame,self.preprocess)
                # frame = asarray(frame)
                #cv2.imshow('frame', frame); cv2.waitKey(10) 
                name = './data/frame' + str(count) + '.jpg'
                cv2.imwrite(name, frame)
                count += 1
            else:
                break
        import ipdb ; ipdb.set_trace()
        vid_tensor = tl.tensor(frame_array)
        vid_tensor = torch.from_numpy(vid_tensor).type(torch.Tensor)
        vid_tensor = vid_tensor.permute(0,3,1,2)
        aug = AugMix()
        vid_tensor2 = aug(vid_tensor)
        cap.release()
        cv2.destroyAllWindows()
        return vid_tensor

    """
    Implementation of AUGMIX as implemented in AUGMIX paper

    """

    CORRUPTIONS = [
        'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
        'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
        'brightness', 'contrast', 'elastic_transform', 'pixelate',
        'jpeg_compression'
    ]


    def aug(self,image, preprocess, op_list):
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
            
            
        # aug_list = augmentations.augmentations_all

        ws = np.float32(np.random.dirichlet([1] * mixture_width))
        m = np.float32(np.random.beta(1, 1))

        mix = torch.zeros_like(self.preprocess(image))
        for i in range(mixture_width):
            image_aug = image.copy()
            depth = mixture_depth if mixture_depth > 0 else np.random.randint(
                1, 4)
            for _ in range(depth):
                # op = np.random.choice(aug_list)  # shear_x, shear_y, translate_x, translate_y doesn't work????
                op = self.aug_list[op_list[i]]
                image_aug = op(image_aug, aug_severity)
            # Preprocessing commutes since all coefficients are convex
            mix += ws[i] * self.preprocess(image_aug)

        mixed = (1 - m) * self.preprocess(image) + m * mix
        mixed = transforms.ToPILImage()(mixed)
        return mixed

    def images_to_video(self, vid_no, frames_dir, fps, image_scale_factor, image_format):
        """
            frames_dir: folder in which images are present
            fps: frame rate (frames per second) for the output video 
            image_scale_factor: Factor by which images is resized. =1 if we dont want to resize 
            image_format: Format of images: .png or jpg (or other -> not tested)
        """
        # Getting all frames
        frame_dir = self.frames_dir + "/" + f"*{image_format}"
        frames = glob.glob(frame_dir)
        frames = natsorted(frames)

        # Get images size
        # import ipdb; ipdb.set_trace()
        img = cv2.imread(frames[0])
        height, width, channels = img.shape
        size = (int(width*image_scale_factor), int(height*image_scale_factor))

        # Video writer Object
        video_out_dir = "/home/patel/mitral/NN-MitralSeg/Augmentation/Echo_data/EchoNet-Dynamic/Synthetic_video/" + "video_" + str(vid_no) + ".avi"
        out = cv2.VideoWriter(video_out_dir, cv2.VideoWriter_fourcc('M','J','P','G'), fps, size)

        for frame in tqdm(frames):
            img = cv2.imread(frame)
            img = cv2.resize(img, (0,0), fx=image_scale_factor, fy=image_scale_factor)
            out.write(img)
        
        out.release()
        print("Video saved to: ", video_out_dir)
        print("Completed")
        shutil.rmtree('/home/patel/mitral/NN-MitralSeg/Augmentation/data')

        return
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