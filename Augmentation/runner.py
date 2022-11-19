from synthetic_data import syn_data
import augmentations

data_folder = "/home/patel/mitral/NN-MitralSeg/Augmentation/Echo_data/EchoNet-Dynamic/Videos"
dir_path = "/home/patel/mitral/NN-MitralSeg/Augmentation/Echo_data/EchoNet-Dynamic"

synthdt = syn_data(data_folder,dir_path)
synthdt.syn_video()
