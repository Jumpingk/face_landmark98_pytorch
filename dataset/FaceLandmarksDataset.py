# -*- coding: utf-8 -*-
"""
Data Loading and Processing Tutorial
====================================
**Author**: `Sasank Chilamkurthy <https://chsasank.github.io>`_

A lot of effort in solving any machine learning problem goes in to
preparing the data. PyTorch provides many tools to make data loading
easy and hopefully, to make your code more readable. In this tutorial,
we will see how to load and preprocess/augment data from a non trivial
dataset.

To run this tutorial, please make sure the following packages are
installed:

-  ``scikit-image``: For image io and transforms
-  ``pandas``: For easier csv parsing

"""

# from __future__ import print_function, division
import random
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import cv2
# from torchvision import transforms,utils
# Ignore warnings
import warnings


######################################################################
# The dataset we are going to deal with is that of facial pose.
# This means that a face is annotated like this:
#
# .. figure:: /_static/img/landmarked_face2.png
#    :width: 400
#
# Over all, 68 different landmark points are annotated for each face.
#
# .. note::
#     Download the dataset from `here <https://download.pytorch.org/tutorial/faces.zip>`_
#     so that the images are in a directory named 'faces/'.
#     This dataset was actually
#     generated by applying excellent `dlib's pose
#     estimation <http://blog.dlib.net/2014/08/real-time-face-pose-estimation.html>`__
#     on a few images from imagenet tagged as 'face'.
#
# Dataset comes with a csv file with annotations which looks like this:
#
# ::
#
#     image_name,part_0_x,part_0_y,part_1_x,part_1_y,part_2_x, ... ,part_67_x,part_67_y
#     0805personali01.jpg,27,83,27,98, ... 84,134
#     1084239450_e76e00b7e7.jpg,70,236,71,257, ... ,128,312
#
# Let's quickly read the CSV and get the annotations in an (N, 2) array where N
# is the number of landmarks.
#

######################################################################
# Let's write a simple helper function to show an image and its landmarks
# and use it to show a sample.
#
# 展示在image上的landmarks图像
def show_landmarks(image, landmarks):
    """Show image with bounding box and landmarks"""
    landmarks = np.array([[landmarks[i], landmarks[i+1]]for i in range(0, len(landmarks), 2)])
    plt.imshow(image)
    # plt.gca().add_patch(plt.Rectangle(xy=(box[0], box[1]), width=box[2]-box[0], height=box[3]-box[1], linewidth=2, edgecolor='g', fill=False))
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=8, marker='.', c='r')
    plt.pause(0.8)  # pause a bit so that plots are updated
    plt.close()


def draw_landmarks(image, landmarks):
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, point in enumerate(landmarks):
        cv2.circle(image, (int(point[0]), int(point[1])), 0, (255, 0, 0), 4)
        cv2.putText(image, str(i), (int(point[0])+2, int(point[1])+2), font, 0.3, (0, 0, 255), 1)
    return image


def OSK(pred_value, truth_value, s, k):
    '''关键点预测评价机制OSK
       s为物体像素面积，因为这里关键点坐标是已经归一化的数据，所以s=1
       k为关键点影响因子
    '''
    norm_ = np.linalg.norm((pred_value.cpu() - truth_value.cpu()).detach().numpy(), ord=2, axis=1, keepdims=True)
    return np.exp(- np.power(norm_, 2) / (2 * np.power(s, 2) * np.power(k, 2)))

######################################################################
# Dataset class
# -------------
#
# ``torch.utils.data.Dataset`` is an abstract class representing a
# dataset.
# Your custom dataset should inherit ``Dataset`` and override the following
# methods:
#
# -  ``__len__`` so that ``len(dataset)`` returns the size of the dataset.
# -  ``__getitem__`` to support the indexing such that ``dataset[i]`` can
#    be used to get :math:`i`\ th sample
#
# Let's create a dataset class for our face landmarks dataset. We will
# read the csv in ``__init__`` but leave the reading of images to
# ``__getitem__``. This is memory efficient because all the images are not
# stored in the memory at once but read as required.
#
# Sample of our dataset will be a dict
# ``{'image': image, 'landmarks': landmarks}``. Our datset will take an
# optional argument ``transform`` so that any required processing can be
# applied on the sample. We will see the usefulness of ``transform`` in the
# next section.
#

class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, img_root, transform=None,rgb = True):
        """
        Args:
            csv_file (s ring): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file, header=None)
        #self.images = np.zeros()
        self.img_root = img_root
        print(len(self.landmarks_frame))
        '''
        for i in xrange(len(self.landmarks_frame)):
            img_name = os.path.join(self.root_dir, self.landmarks_frame.ix[i, 0])
            image = cv2.imread(img_name)
            self.img_list.append(image)
        '''
        self.transform = transform
        self.rgb = rgb

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_root, self.landmarks_frame.loc[idx, 0])
        image = cv2.imread(img_name)
        if self.rgb:  # 表示是否要转换为rgb图像
            image = image[...,::-1]  # 第三个通道与第一个通道互换，转换为RGB图像, 因为cv2读取的图像为bgr图像，需要进行转换
        landmarks = self.landmarks_frame.loc[idx, 1:].values.astype('float') 
        landmarks = landmarks.reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample


######################################################################
# Let's instantiate this class and iterate through the data samples. We
# will print the sizes of first 4 samples and show their landmarks.
#


######################################################################
# Transforms
# ----------
#
# One issue we can see from the above is that the samples are not of the
# same size. Most neural networks expect the images of a fixed size.
# Therefore, we will need to write some prepocessing code.
# Let's create three transforms:
#
# -  ``Rescale``: to scale the image
# -  ``RandomCrop``: to crop from image randomly. This is data
#    augmentation.
# -  ``ToTensor``: to convert the numpy images to torch images (we need to
#    swap axes).
#
# We will write them as callable classes instead of simple functions so
# that parameters of the transform need not be passed everytime it's
# called. For this, we just need to implement ``__call__`` method and
# if required, ``__init__`` method. We can then use a transform like this:
#
# ::
#
#     tsfm = Transform(params)
#     transformed_sample = tsfm(sample)
#
# Observe below how these transforms had to be applied both on the image and
# landmarks.
#

class SmartRandomCrop(object):
    """Crop randomly the image in a sample.

    # SmartRandomCrop类的作用是：
    # 获得最小关键点包围框zoom_scale倍扩边之后的裁剪图片和关键点相对于裁剪图片的坐标位置

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, zoom_scale=1.5):  # zoom scale : 缩放尺度
        assert isinstance(zoom_scale, (int, float))
        self.zoom_scale = zoom_scale
    def get_random_rect(self,min_x,min_y,max_x,max_y,w,h):
        rec_w = max_x  - min_x
        rec_h = max_y  - min_y
        scale = (self.zoom_scale-1)/2.0
        b_min_x = min_x - rec_w*scale if min_x - rec_w*scale >0 else 0
        b_min_y = min_y - rec_h*scale if min_y - rec_h*scale >0 else 0
        b_max_x = max_x + rec_w*scale if max_x + rec_w*scale <w else w
        b_max_y = max_y + rec_h*scale if max_y + rec_h*scale <h else h
        #r_min_x = np.random.randint(int(b_min_x),int(min_x)) if b_min_x<min_x else int(min_x)
        #r_min_y = np.random.randint(int(b_min_y),int(min_y)) if b_min_y<min_y else int(min_y)
        #r_max_x = np.random.randint(int(max_x),int(b_max_x)) if b_max_x > max_x else int(max_x)
        #r_max_y = np.random.randint(int(max_y),int(b_max_y)) if b_max_y > max_y else int(max_y)
        return b_min_x,b_min_y,b_max_x,b_max_y

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        min_xy  = np.min(landmarks,axis= 0)
        max_xy  = np.max(landmarks,axis= 0)
        min_x,min_y,max_x,max_y = self.get_random_rect(min_xy[0],min_xy[1],max_xy[0],max_xy[1],w,h)
        image = image[int(min_y): int(max_y),
                int(min_x):int(max_x)]

        landmarks = landmarks - [min_x, min_y]

        return {'image': image, 'landmarks': landmarks}

class Rescale(object):
    """Rescale the image in a sample to a given size.
    将样本中的图像缩放到给定的大小
    Args:
        output_size (tuple or tuple): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        #img = transform.resize(image, (new_h, new_w))
        img = cv2.resize(image,(new_w,new_h))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'landmarks': landmarks}


class RandomFlip(object):
    """对图像进行1/2概率的水平翻转"""
    def __call__(self, sample):
        image = sample['image']
        landmarks = sample['landmarks']
        if random.random()<0.5:
            image = cv2.flip(image,1)  # 1 水平翻转
            landmarks[:,0] = image.shape[1]-landmarks[:,0]
        return {'image': image, 'landmarks': landmarks}


class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image
class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, sample):
        image = sample['image']
        if random.randint(0,2):  # 2/3的概率
            swap = self.perms[random.randint(0,len(self.perms)-1)]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
            sample['image'] = image
        return sample

class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, sample):
        if random.randint(0,2):
            image = sample['image']
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
            sample['image'] = image
        return sample

class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, sample):
        image = sample['image']
        if random.randint(0,2):
            delta = random.uniform(-self.delta, self.delta)
            np.add(image, delta, out=image, casting="unsafe")
            #image += delta
            sample['image'] = image
        return sample


# class RandomCrop(object):
#     """Crop randomly the image in a sample.

#     Args:
#         output_size (tuple or int): Desired output size. If int, square crop
#             is made.
#     """

#     def __init__(self, output_size):
#         assert isinstance(output_size, (int, tuple))
#         if isinstance(output_size, int):
#             self.output_size = (output_size, output_size)
#         else:
#             assert len(output_size) == 2
#             self.output_size = output_size

#     def __call__(self, sample):
#         image, landmarks = sample['image'], sample['landmarks']

#         h, w = image.shape[:2]
#         new_h, new_w = self.output_size

#         top = np.random.randint(0, h - new_h)
#         left = np.random.randint(0, w - new_w)

#         image = image[top: top + new_h,
#                 left: left + new_w]

#         landmarks = landmarks - [left, top]

#         return {'image': image, 'landmarks': landmarks}


      
class ToTensor(object):
    """Convert ndarrays in sample to Tensors.
    并且对图像矩阵和关键点坐标进行了归一化处理
    """
    def __init__(self,image_size):
        self.image_size = image_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        landmarks =landmarks.reshape(-1,1)
        image = torch.from_numpy(image).float().div(255)  # 修改
        # image = torch.from_numpy(image).float()
        landmarks = torch.from_numpy(landmarks).float().div(self.image_size)
        return {'image': image.clamp(0, 1),  #    # 修改
                'landmarks': landmarks.clamp(0, 1)} 


class Normalize(object):
    def __init__(self,mean,std):
        self.mean = mean
        self.std = std
    def __call__(self, sample):
        image = sample['image']
        for t, m, s in zip(image, self.mean, self.std):
            t.sub_(m).div_(s)
        sample['image'] = image
        return sample

if __name__ == "__main__":
    import torchvision.transforms as transforms
    import torch.utils.data as data
    from tqdm import tqdm

    transform_train = transforms.Compose([

        SmartRandomCrop(zoom_scale=1.5),
        Rescale((224, 224)),
        # RandomCrop((64,64)),
        RandomFlip(),
        #RandomContrast(),
        RandomBrightness(),
        RandomLightingNoise(),
        ToTensor(224),
        # Normalize([ 0.485, 0.456, 0.406 ],
        #                     [ 0.229, 0.224, 0.225 ]),
    ])

    trainSet = FaceLandmarksDataset(
        csv_file='/home/cupk/document/vscode_python/pytorch_face_landmark/data/face_landmark_train.csv',
        img_root='/home/cupk/data/WFLW_images',
        transform=transform_train
    )
    # print(trainSet)
    get_some_data = np.array([trainSet[i]['image'].numpy() for i in range(1000)])
    # for k in range(500):
    #     data.append(trainSet[k]['image'].numpy())
    print(get_some_data.shape)
    # 计算部分数据的均值和方差
    print('mean: ', np.mean(np.mean(np.mean(get_some_data, axis=0), axis=1), axis=1))
    print('std: ', np.mean(np.mean(np.std(get_some_data, axis=0), axis=1), axis=1))
    exit()
    trainLoader = data.DataLoader(dataset=trainSet, batch_size=1, shuffle=True, num_workers=0)
    print(len(trainLoader))
    for x in trainLoader:
        print(x['image'].shape)
        print(x['landmarks'].shape)
    








######################################################################
# Compose transforms
# ~~~~~~~~~~~~~~~~~~
#
# Now, we apply the transforms on an sample.
#
# Let's say we want to rescale the shorter side of the image to 256 and
# then randomly crop a square of size 224 from it. i.e, we want to compose
# ``Rescale`` and ``RandomCrop`` transforms.
# ``torchvision.transforms.Compose`` is a simple callable class which allows us
# to do this.
#

'''
scale = Rescale(256)
crop = RandomCrop(128)
composed = transforms.Compose([Rescale(256),
                               RandomCrop(224)])

# Apply each of the above transforms on sample.
fig = plt.figure()
sample = face_dataset[65]
for i, tsfrm in enumerate([scale, crop, composed]):
    transformed_sample = tsfrm(sample)

    ax = plt.subplot(1, 3, i + 1)
    plt.tight_layout()
    ax.set_title(type(tsfrm).__name__)
    show_landmarks(**transformed_sample)

plt.show()
'''
