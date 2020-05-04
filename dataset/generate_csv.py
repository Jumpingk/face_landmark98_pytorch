'''
从原始数据中生成csv文件
'''
import cv2
import os
import random
from tqdm import tqdm
anno_root = '/home/cupk/data/WFLW_annotations/list_98pt_rect_attr_train_test' # 原始数据的标签文件夹
img_root = '/home/cupk/data/WFLW_images' # 原始数据图片文件夹
data_store = '../data'
train_rate = 1.0


if __name__ == "__main__":
    # 98关键点数据读取，生成训练集、验证集、测试集csv文件
    items = []
    # (train data) and (val data) and (test data)
    with open(os.path.join(anno_root, 'list_98pt_rect_attr_train.txt'), 'r') as train_val_items:
        for x in train_val_items.readlines():
            all_info = x.strip().split(' ')
            if len(all_info) == 207:
                items.append(all_info[-1] + ',' + ','.join(all_info[:196]) + '\n')
            else:
                print(all_info[-1])
    IndexSplit = int(len(items) * train_rate)
    train_items = items[:IndexSplit]
    val_items = items[IndexSplit:]
    test_items = []
    with open(os.path.join(anno_root, 'list_98pt_rect_attr_test.txt'), 'r') as test_items_:
        for y in test_items_.readlines():
            test_info = y.strip().split(' ')
            if len(test_info) == 207:
                test_items.append(test_info[-1] + ',' + ','.join(test_info[:196]) + '\n')
            else:
                print(test_info[-1])

    # title info
    title_info = ['image_index']
    for i in range(0, 98):
        for j in ['x', 'y']:
            title_info.append('point_' + j + str(i))

    # ###### 表示是否要加表头信息 ##############
    # csv_title = ','.join(title_info) + '\n'
    csv_title = None

    if not os.path.exists(data_store):
        os.mkdir(data_store)
    with open(os.path.join(data_store, 'face_landmark_train.csv'), 'w') as trainfile:
        print('generate face lanmark train csv file ...')
        if csv_title is not None:
            trainfile.write(csv_title)
        train_bar = tqdm(train_items)
        for item in train_bar:
            trainfile.write(item)
    with open(os.path.join(data_store, 'face_landmark_val.csv'), 'w') as valfile:
        print('generate face landmark val csv file ...')
        if csv_title is not None:
            valfile.write(csv_title)
        val_bar = tqdm(val_items)
        for item in val_bar:
            valfile.write(item)
    with open(os.path.join(data_store, 'face_landmark_test.csv'), 'w') as testfile:
        print('generate face landmark test csv file ...')
        if csv_title is not None:
            testfile.write(csv_title)
        test_bar = tqdm(test_items)
        for item in test_bar:
            testfile.write(item)

    # generate data info
    with open(os.path.join(data_store, 'data_info.txt'), 'w') as infofile:
        infofile.write('训练集、验证集中，训练集所占比例为：{}\n'.format(train_rate))
        infofile.write('| 训练集 | 验证集 | 测试集 |\n')
        infofile.write('| ' + str(len(train_items)) + '  | ' + str(len(val_items)) + '  | ' + str(len(test_items)) + '  |\n')

    # read data info
    with open(os.path.join(data_store, 'data_info.txt'), 'r') as info:
        for i in info.readlines():
            print(i)

