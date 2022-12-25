import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from numpy import asarray
import augmentations
import pickle
preprocess = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)])


def aug(image, preprocess):
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

        mix = torch.zeros_like(preprocess(image))
        for i in range(mixture_width):
            image_aug = image.copy()
            depth = mixture_depth if mixture_depth > 0 else np.random.randint(
                1, 4)
            for _ in range(depth):
               #op = np.random.choice(aug_list)
                op = aug_list[10]
                image_aug = op(image_aug, aug_severity)
            # Preprocessing commutes since all coefficients are 
            mix += ws[i] * preprocess(image_aug)

        mixed = (1 - m) * preprocess(image) + m * mix 
        mixed = transforms.ToPILImage()(mixed)
        return mixed

### Augmenting the video 


data_folder = "/home/patel/mitral/NN-MitralSeg/Augmentation/Echo_data/EchoNet-Dynamic/Videos"
dir_path = "/home/patel/mitral/NN-MitralSeg/Augmentation/Echo_data/EchoNet-Dynamic"
video_list = os.listdir(os.path.join(dir_path, str(data_folder)))
#import ipdb; ipdb.set_trace()
for i, vid in enumerate(video_list):
    video_list[i] = os.path.join( str(data_folder),vid)

print(video_list)
# for i in range(len(video_list)):

        
#         #dt = load_zipped_pickle(os.path.join(dir_path, str(data_folder), video_list[i]))
#         with open(os.path.join(dir_path, str(data_folder), video_list[i]), 'rb') as f:
#             dt = pickle.load(f)

video = video_list[2]
cap = cv2.VideoCapture(video)
count = 0

try:
      
    # creating a folder named data
    if not os.path.exists('data'):
        os.makedirs('data')
  
# if not created then raise error
except OSError:
    print ('Error: Creating directory of data')

    
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # cv2.imshow('frame', frame); cv2.waitKey(10)
        frame = transforms.ToPILImage()(frame)
        frame = aug (frame,preprocess)
        frame = asarray(frame)
        #cv2.imshow('frame', frame); cv2.waitKey(10)
        #cv2.imwrite("frame%d.jpg" % count, frame) 
        name = './data/frame' + str(count) + '.jpg'
        print ('Creating...' + name)
        cv2.imwrite(name, frame)
        count += 1
    else:
        break
cap.release()
cv2.destroyAllWindows()