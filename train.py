'''
Training script for CIFAR-10/100
Copyright (c) Wei YANG, 2017
'''
from __future__ import print_function

import argparse
import os
import shutil
import time
import random

import torch
import torchvision
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.tensorboard import SummaryWriter
from models import *
from dataset.FaceLandmarksDataset import *

from utils import Bar, Logger, AverageMeter, normalizedME, mkdir_p, savefig
print('import OK')

parser = argparse.ArgumentParser(description='PyTorch face landmark Training')
# Datasets  # 准备数据集
parser.add_argument('-d', '--dataset', default='face98', type=str)
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=120, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',  # 设置模型开始训练的节点，可以在以前训练的基础上继续训练
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=256, type=int, metavar='N', # train_batch
                    help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[90],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint/1011/', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
#parser.add_argument('--resume', default='/home/foto1/workspace/zuoxin/face_landmark/checkpoint/0918/facelandmark_squeezenet_128_55.pth.tar', type=str, metavar='PATH',
#                    help='path to latest checkpoint (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--depth', type=int, default=104, help='Model depth.')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed', default=1, type=int, help='manual seed')  # 表示是否要设立随机种子，保证每次训练的初始化保持一致
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
#Device options
parser.add_argument('--gpu_id', default='0', type=str,  # 也可以进行多个GPU设备的设置，例如：default='0,1,2'
help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}
# print(state)
# exit()


# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id  # 设置使用的GPU设备
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    # torch.cuda.manual_seed(args.seed)#为当前GPU设置随机种子；如果使用多个GPU，应该使用torch.cuda.manual_seed_all()为所有的GPU设置种子
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0.  # best test accuracy

def main():
    global best_acc  # 设置全局变量best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch  # 设置模型训练开始的节点

    if not os.path.isdir(args.checkpoint):  # 如果不存在目录checkpoint，就创建该目录
        mkdir_p(args.checkpoint)




    # Data  # 数据预处理操作
    print('==> Preparing dataset %s' % args.dataset)
    transform_train = transforms.Compose([
        SmartRandomCrop(zoom_scale=1.5),
        Rescale((224, 224)),
        # RandomCrop((64,64)),
        RandomFlip(),
        #RandomContrast(),
        RandomBrightness(),
        RandomLightingNoise(),
        ToTensor(224),   # 修改
        Normalize([0.4705, 0.4384, 0.4189],
                          [0.2697, 0.2621, 0.2662]), # 修改
    ])

    transform_test = transforms.Compose([
        SmartRandomCrop(zoom_scale=1.5),
        Rescale((224, 224)),
        ToTensor(224),  # 修改
        Normalize([0.4705, 0.4384, 0.4189],
                           [0.2697, 0.2621, 0.2662]), # 修改
    ])

    trainset = FaceLandmarksDataset(
        csv_file='/home/cupk/document/vscode_python/pytorch_face_landmark/data/face_landmark_train.csv',
        img_root='/home/cupk/data/WFLW_images',
        transform=transform_train
    )
    trainloader = data.DataLoader(
        dataset=trainset,
        batch_size=args.train_batch,
        shuffle=True,
        num_workers=args.workers
    )

    testset = FaceLandmarksDataset(
        csv_file='/home/cupk/document/vscode_python/pytorch_face_landmark/data/face_landmark_test.csv',
        img_root='/home/cupk/data/WFLW_images',
        transform=transform_test
    )
    testloader = data.DataLoader(
        dataset=testset,
        batch_size=args.test_batch,
        shuffle=True,
        num_workers=args.workers
    )
    # model = AlexNet(196)
    # model = ImprovedAlexNet(196)
    # model = MobileNetV2(196)
    model = SqueezeNet(196)

    writer = SummaryWriter('checkpoint/SqueezeNet')
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    # criterion = nn.MSELoss(reduction='sum').cuda()  # MSE
    criterion = nn.SmoothL1Loss(reduction='sum').cuda() # Huber 损失函数  # 修改
    # criterion = nn.L1Loss(reduction='sum').cuda()  # MAE
    
    
    ignored_params = list(map(id, model.fc.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params,
                         model.parameters())
    params = [
        {'params': base_params, 'lr': args.lr},
        {'params': model.fc.parameters(), 'lr': args.lr * 10}
    ]
    model = torch.nn.DataParallel(model).cuda()  # 可以使模型在多个GPU上进行训练
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = optim.Adam(params=params, lr=args.lr, weight_decay=args.weight_decay)

    # Resume
    title = 'facelandmark_Modi_SqueezeNet_224' # 修改 （一）
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if os.path.exists(os.path.join(args.checkpoint, title+'_log.txt')):
            logger = Logger(os.path.join(args.checkpoint, title+'_log.txt'), title=title, resume=True)
        else:
            logger = Logger(os.path.join(args.checkpoint, title+'_log.txt'), title=title)
            logger.set_names(['Learning Rate', 'Train Loss', 'Test Loss', 'Train Acc[%].', 'Test Acc[%].'])
    else:
        logger = Logger(os.path.join(args.checkpoint, title+'_log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Test Loss', 'Train Acc[%].', 'Test Acc[%].'])


    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(testloader, model, criterion, start_epoch, use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return

    # Train and val
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss, train_acc = train(trainloader, model, criterion, optimizer, epoch, use_cuda, writer)
        test_loss, test_acc = test(testloader, model, criterion, epoch, use_cuda, writer)

        # append logger file
        logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc])

        # save model
        is_best = test_acc > best_acc
        best_acc =max(test_acc, best_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint=args.checkpoint,filename=title+'_'+str(epoch)+'.pth.tar')

    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps'))
    writer.close()

    print('Best acc:')
    print(best_acc)

def train(trainloader, model, criterion, optimizer, epoch, use_cuda, writer):
    # switch to train mode
    model.train()
    # AverageMeter 为pytorch中自定义的平均损失计算函数
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    NormMS = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(trainloader))
    scores = 0.
    count_num = 0
    for batch_idx, batch_data in enumerate(trainloader):
        count_num += 1
        # measure data loading time
        data_time.update(time.time() - end)
        inputs = batch_data['image']
        targets = batch_data['landmarks']
        img_grid = torchvision.utils.make_grid(inputs)
        # if (epoch + 1) % 20 == 0:
        #     writer.add_image('sixteen_face_images', img_grid)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(async=True)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        # print('input_size: ', inputs.size())
        outputs = model(inputs)
        score = np.mean(OSK(outputs, targets.squeeze(), 1, 1))
        scores += score
        loss = criterion(outputs, targets.squeeze())

        # measure accuracy and record loss
        #nms= normalizedME(outputs.data,targets.data,64,64)
        losses.update(loss.item(), inputs.size(0))
        #NormMS.update(nms[0], inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(float(time.time() - end))
        end = time.time()
        writer.add_scalars('train', {'loss': losses.avg, 'acc[%]': score*100}, epoch*len(trainloader) + batch_idx)

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, scores*100/count_num)

def test(testloader, model, criterion, epoch, use_cuda, writer):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(testloader))
    scores = 0.
    count_num = 0
    for batch_idx, batch_data in enumerate(testloader):
        count_num += 1
        # measure data loading time
        data_time.update(time.time() - end)

        inputs = batch_data['image']
        targets = batch_data['landmarks']
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(async=True)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
        
        # compute output
        outputs = model(inputs)
        score = np.mean(OSK(outputs, targets.squeeze(), 1, 1))
        scores += score
        loss = criterion(outputs, targets.squeeze())

        # measure accuracy and record loss
        losses.update(loss.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(float(time.time() - end))
        end = time.time()
        writer.add_scalars('test', {'loss': losses.avg, 'acc[%]': score*100}, epoch*len(testloader) + batch_idx)

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
    return (losses.avg, scores*100/count_num)

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch):  # 根据schedule和gamma这两个参数动态调整学习率
    global state
    if epoch+1 in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

if __name__ == '__main__':
    main()
