import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import lycon
from dataset.FaceLandmarksDataset import *
from models import *








class ModelInference:
    def __init__(self, 
            image_size:int,  # image_s
            network, 
            pth_model_path:str, 
            # use_cuda=True, 
            rgb=False,
            ):

        self.model = network
        self.checkpoint = torch.load(pth_model_path)
        print('epoch: ', self.checkpoint['epoch'])
        print('acc: ', self.checkpoint['acc'])
        print('best_acc: ', self.checkpoint['best_acc'])
        self.model.load_state_dict({k.replace('module.', ''):v for k, v in self.checkpoint['state_dict'].items()}) # ['state_dict']
        # self.optimizer = optim.Adam()
        # self.optimizer.load_state_dict(checkpoint['optimizer'])
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.image_size = image_size
        self.rgb = rgb
        self.model.eval()
        self.mean = [0.4705, 0.4384, 0.4189]
        self.std = [0.2697, 0.2621, 0.2662]
    
    def pre_processing(self, img_matrix):
        image = lycon.resize(img_matrix, self.image_size, self.image_size)
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image).float().div(255) # 
        image = image.clamp(0, 1)
        for t, m, s in zip(image, self.mean, self.std):
            t.sub_(m).div_(s)
        return image

    def pic_inference(self, img_path):
        '''input image/images is/are face picture/pictures.
           传入图片路径或图片路径列表
        '''
        if isinstance(img_path, str):
            image = lycon.load(img_path)
            if self.rgb:  # 表示是否要转换为rgb图像(cv2要转rgb图像)
                image = image[...,::-1]
            base_image = image
            img_h, img_w = image.shape[:2]
            image = self.pre_processing(image).unsqueeze(0)
            print(image.shape)
            # exit()
            landmarks = self.model(image.cuda())

            landmarks = landmarks.cpu().detach().numpy().reshape(-1, 2)
            new_landmarks = []
            for h, w in landmarks:
                new_landmarks.append([h*img_h, w*img_w])
            base_image = cv2.cvtColor(base_image, cv2.COLOR_BGR2RGB)
            img = draw_landmarks(base_image, new_landmarks)
            cv2.imshow('img', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()



        elif isinstance(img_path, list):
            for img in img_path:
                image = lycon.load(img)
                if self.rgb:  # 表示是否要转换为rgb图像(cv2要转rgb图像)
                    image = image[...,::-1]
                base_image = image
                img_h, img_w = image.shape[:2]
                image = self.pre_processing(image)
                
        else:
            raise(NameError('input error!!!'))
        

        

        pass
    def video_inference():
        pass


if __name__ == "__main__":
    img_path = "/home/cupk/document/vscode_python/landmark_test_pic/pic_4520.jpg" # pic_1960.jpg  4520  0  4400  4720

    inferencer = ModelInference(
        image_size=224,
        network=SqueezeNet(196).cuda(),  # 196
        pth_model_path='/home/cupk/document/vscode_python/pytorch_face_landmark/checkout/SqueezeNet/facelandmark_Modi_SqueezeNet_224_129.pth.tar', # model_best.pth  facelandmark_SqueezeNet_224_158
        rgb=False)  # facelandmark_Modi_SqueezeNet_224_129.pth
    inferencer.pic_inference(img_path)
    