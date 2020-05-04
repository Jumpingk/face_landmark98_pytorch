from __future__ import division
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import tqdm
# img_root = '/home/cupk/data/WFLW_images'
# dst_root = '/home/cupk/data/WFLW_cropped_images_2'
# landmarks_frame = pd.read_csv('../data/face_landmark_val.csv')
# print(landmarks_frame.ix[:, 0])
# print('length: {}'.format(len(landmarks_frame)))
# new_img_root = '../data/processed_data'








# SmartRandomCrop类的作用是：
# 获得最小关键点包围框1.5倍扩边之后的裁剪图片和关键点相对于裁剪图片的坐标位置
class SmartRandomCrop(object):
    """Crop randomly the image in a sample.

    # SmartRandomCrop类的作用是：
    # 获得最小关键点包围框1.5倍扩边之后的裁剪图片和关键点相对于裁剪图片的坐标位置

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, zoom_scale = 3):  # zoom scale : 缩放尺度
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

def show_landmarks(image, landmarks, save_fig=False, save_path=None):
    """Show image with landmarks"""

    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    if save_fig and save_path is not None:
        plt.savefig(save_path)
    plt.pause(0.001)  # pause a bit so that plots are updated

def draw_landmarks(image, landmarks):
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, point in enumerate(landmarks):
        cv2.circle(image, (int(point[0]), int(point[1])), 0, (255, 0, 0), 4)
        cv2.putText(image, str(i), (int(point[0])+2, int(point[1])+2), font, 0.3, (0, 0, 255), 1)
    return image

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


def get_some_pictures(pic_seq):
    '''从训练集中获取一部分图片以及标好关键点的图片
    '''
    random_crop = SmartRandomCrop(1.5)
    rescale = Rescale((448, 448))
    store_photos = '/home/cupk/document/vscode_python/landmark_test_pic'
    img_root = '/home/cupk/data/WFLW_images'
    with open('/home/cupk/document/vscode_python/pytorch_face_landmark/data/face_landmark_train.csv', 'r') as f:
        infos = f.readlines()
        for k in tqdm(pic_seq):
            info_list = infos[k].strip().split(',')
            image_path = os.path.join(img_root, info_list[0])
            if os.path.exists(image_path):
                image = cv2.imread(image_path)
                landmarks = np.array([float(x) for x in info_list[1:]])
            else:
                raise FileNotFoundError(image_path + ' NOT FOUND THIS FILE!')
            landmarks = landmarks.reshape(-1, 2)
            sample = {'image': image, 'landmarks': landmarks}
            crop_info = random_crop(sample)
            rescale_info = rescale(crop_info)
            new_img = rescale_info['image']
            new_landmarks = rescale_info['landmarks']
            # cv2.imwrite(os.path.join(store_photos, 'pic' + '_' + str(k) + '.jpg'), new_img, [cv2.IMWRITE_JPEG_QUALITY, 100])
            print(new_landmarks.shape)
            landmark_img = draw_landmarks(new_img, new_landmarks)
            cv2.imshow('landmark_img', landmark_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            # cv2.imwrite(os.path.join(store_photos, 'pic' + '_' + str(k) + '_landmarks' + '.jpg'), landmark_img, [cv2.IMWRITE_JPEG_QUALITY, 100])


def store_all_face_pictures():
    random_crop = SmartRandomCrop(1.5)
    rescale = Rescale((448, 448))
    unprocessed_data = '../data/unprocessed_data'
    if not os.path.exists('../data/processed_data'):
        os.mkdir('../data/processed_data')
    for attribute in ['train', 'test', 'val']:
        processed_data = '../data/processed_data/' + attribute + '_data'
        if not os.path.exists(processed_data):
            os.mkdir(processed_data)
        new_image_path = os.path.join(processed_data, 'image')
        if not os.path.exists(new_image_path):
            os.mkdir(new_image_path)
        
        with open(os.path.join(processed_data, attribute + '_label.txt'), 'w') as nfile:
            with open(os.path.join(unprocessed_data, 'face_landmark_' + attribute + '.csv'), 'r') as f:
                for pic_info in f.readlines():
                    info_list = pic_info.strip().split(',')
                    image_path = info_list[0]
                    print(image_path)
                    image = cv2.imread(image_path)
                    if os.path.exists(image_path):
                        landmarks = np.array([float(x) for x in info_list[1:]])
                    else:
                        raise FileNotFoundError(image_path + ' NOT FOUND THIS FILE!')
                    landmarks = landmarks.reshape(-1, 2)
                    sample = {'image': image, 'landmarks': landmarks}
                    crop_info = random_crop(sample)
                    rescale_info = rescale(crop_info)
                    new_img = rescale_info['image']
                    new_landmarks = rescale_info['landmarks']
                    new_img_path = os.path.join(new_image_path, str(int(1000000*time.time()))[-10:] + '_' + os.path.basename(image_path))
                    cv2.imwrite(new_img_path, new_img)
                    nfile.write(new_img_path)
                    nfile.write(',' + ','.join([str(x) for x in new_landmarks]) + '\n')


if __name__ == "__main__":
    get_some_pictures([i for i in range(0, 5000, 40)])














# random_crop = SmartRandomCrop(1.5) # 智能随机裁剪
# rescale = Rescale((128,128))
# if not os.path.exists(dst_root):
#     os.makedirs(dst_root)

# ofile = open('cropped_face_landmarks_val_2.txt','w')
# for idx in range(len(landmarks_frame)):
#     img_name = os.path.join(img_root, landmarks_frame.ix[idx, 0])
#     print(img_name)
#     image = cv2.imread(img_name)
#     landmarks = landmarks_frame.ix[idx, 1:].as_matrix().astype('float')
#     landmarks = landmarks.reshape(-1, 2)
#     sample = {'image':image,'landmarks':landmarks}
#     sample = random_crop(sample)
#     cv2.imwrite(os.path.join(dst_root,landmarks_frame.ix[idx,0]),sample['image'])
#     landmarks = sample['landmarks']
#     img = sample['image']
#     ofile.write(landmarks_frame.ix[idx,0])
#     for i in range(76):
#         ofile.write(','+str(landmarks[i][0])+','+str(landmarks[i][1]))
#     ofile.write('\n')

