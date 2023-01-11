from synthetic_data import syn_data
import augmentations
import cv2

data_folder = "/home/patel/mitral/NN-MitralSeg/Augmentation/Echo_data/EchoNet-Dynamic/Videos"
dir_path = "/home/patel/mitral/NN-MitralSeg/Augmentation/Echo_data/EchoNet-Dynamic"
frames_dir = '/home/patel/mitral/NN-MitralSeg/Augmentation/data'

synthdt = syn_data(data_folder,dir_path, frames_dir)
synthdt.syn_video()
