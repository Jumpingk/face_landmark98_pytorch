import os
import lycon
import cv2
import pandas as pd
import numpy as np
from dataset.FaceLandmarksDataset import show_landmarks


# 展示数据集(keypoints and faces)
def show_data_renderings(img_root, csv_file_path):
    with open(csv_file_path, 'r') as f:
        for pic_info in f.readlines():
            info_list = pic_info.strip().split(',')
            img_path = os.path.join(img_root, info_list[0])
            print(img_path)
            if os.path.exists(img_path):
                landmarks = [float(x) for x in info_list[1:]]
                show_landmarks(lycon.load(img_path), landmarks)
            else:
                raise FileNotFoundError(img_path + 'NOT FOUND THIS FILE!')


# 读取csv文件展示数据
def read_csv_data(img_root, csv_path):
    landmarks_frame = pd.read_csv(csv_path, header=None)
    print(len(landmarks_frame))
    # print(landmarks_frame.loc[0, 0])

    for i in range(len(landmarks_frame)):
        img_name = os.path.join(img_root, landmarks_frame.loc[i, 0])
        landmarks = landmarks_frame.loc[i, 1:].values.astype('float')     #    .as_matrix().astype('float')
        print(type(landmarks))
        print(landmarks)
        exit()
        img = cv2.imread(img_name)
        cv2.imshow('img', img)
        cv2.waitKey(1000)
        cv2.destroyWindow('img')


if __name__ == "__main__":
    # show_data_renderings(img_root='/home/cupk/data/WFLW_images', csv_file_path='/home/cupk/document/vscode_python/pytorch_face_landmark/data/face_landmark_val.csv')
    read_csv_data(img_root='/home/cupk/data/WFLW_images', csv_path='/home/cupk/document/vscode_python/pytorch_face_landmark/data/face_landmark_val.csv')

