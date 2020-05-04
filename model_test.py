'''Test Data to test
(c) DongQi Wang
'''
import os
import numpy as np
import cv2
import time
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from utils import AverageMeter, Bar
from dataset.FaceLandmarksDataset import *
from models import *



def test(testloader, model, criterion, use_cuda):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(testloader))
    for batch_idx, batch_data in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        inputs = batch_data['image']
        targets = batch_data['landmarks']
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(async=True)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets.squeeze())

        # measure accuracy and record loss
        losses.update(loss.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(float(time.time() - end))
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} '.format(
                    batch=batch_idx + 1,
                    size=len(testloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg,0)


def test_model(network, pth_model_path, test_batch, workers):
    '''
    Feed test data into model
    '''
    transform_test = transforms.Compose([
        #SmartRandomCrop(),
        Rescale((224, 224)),
        ToTensor(224),
        Normalize([0.4705, 0.4384, 0.4189],
                            [0.2697, 0.2621, 0.2662]),
    ])

    testset = FaceLandmarksDataset(
        csv_file='/home/cupk/document/vscode_python/pytorch_face_landmark/data/face_landmark_test.csv',
        img_root='/home/cupk/data/WFLW_images',
        transform=transform_test
    )
    testloader = data.DataLoader(
        dataset=testset,
        batch_size=test_batch,
        shuffle=True,
        num_workers=workers
    )
    model = torch.nn.DataParallel(network).cuda()
    checkpoint = torch.load(pth_model_path)
    # print(checkpoint)
    # for c in checkpoint:
    #     print(c)
    # exit()
    model.load_state_dict(checkpoint['state_dict'])
    criterion = nn.MSELoss(reduction='sum')
    use_cuda = True
    test(testloader, model, criterion, use_cuda)
    print('end...')


def post_processing():
    pass

    



def model_inference(image):
    pass


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # network = AlexNet(196)
    # network = ImprovedAlexNet(196)
    # network = MobileNetV2(196)
    network = SqueezeNet(196)
    model_path = './checkout/1011/facelandmark_vgg16_224_0.pth.tar'
    test_batch = 32
    workers = 0
    test_model(network=network, pth_model_path=model_path, test_batch=test_batch, workers=workers)
