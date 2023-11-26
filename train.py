import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.optim import lr_scheduler
from torchvision import models
from torchvision import transforms
import torch.optim as optim
from EntroMixer import AttentionBlock, rank, vfe
import os
import scipy.fftpack as fp
from PIL import ImageFile
from tqdm import *

ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def topk_accuracy(output, targets, k):
    values, indices = torch.topk(output, k=k, sorted=True)
    targets = torch.reshape(targets, [-1, 1])
    correct = (targets == indices) * 1
    top_k_accuracy = torch.sum(torch.max(correct, dim=1)[0])
    return top_k_accuracy

def fft(img):
    fft_image = np.fft.fft2(img)
    shifted_fft_image = np.fft.fftshift(fft_image)
    center_y, center_x = shifted_fft_image.shape[0] // 2, shifted_fft_image.shape[1] // 2
    size = 50
    region = shifted_fft_image[center_y - size // 2:center_y + size // 2, center_x - size // 2:center_x + size // 2]
    high_freq_magnitude = np.mean(np.abs(region))

    return high_freq_magnitude


def train(trainloader, model, rank_moudle, device, optimizer, schedule, criterion):
    train_acc = 0
    train_loss = 0
    model.train()
    rank_moudle.train()
    loop = tqdm((trainloader), total=(len(trainloader)))
    train_sample_num = 0
    for i,(inputs, targets) in enumerate(loop):
        train_sample_num += inputs.shape[0]
        inputs = inputs.to(device)
        target = targets.to(device)
        inputs, channel_w, spatial_w = rank_moudle(inputs)
        for i in range(inputs.shape[0]):
            v, _ = vfe(inputs[i], 14, 14)
            inputs[i:,:,:] = rank(inputs[i:,:,:], channel_w, spatial_w, 14, v, [0.5, 1])
        output = model(inputs)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        num = topk_accuracy(output, target, 1)
        train_loss += loss.item()
        train_acc += num.item()
        loop.set_description(f'Epoch:train [{i}/{len(trainloader)}]')
        loop.set_postfix(loss=train_loss / train_sample_num, acc=train_acc / train_sample_num)
        schedule.step()


def test(model, rank_moudle, testloader, best_acc, device, criterion, model_save_dir):
    model.eval()
    rank_moudle.eval()
    val_sample_num = 0
    loop = tqdm((testloader), total=len(testloader))
    val_loss = 0
    val_acc_num = 0
    val_steps = 0
    for i, (inputs, target) in enumerate(loop):
        with torch.no_grad():
            val_sample_num += inputs.shape[0]
            inputs = inputs.to(device)
            target = target.to(device)
            inputs, channel_w, spatial_w = rank_moudle(inputs)
            for i in range(inputs.shape[0]):
                v, _ = vfe(inputs[i], 14, 14)
                inputs[i:,:,:] = rank(inputs[i:,:,:], channel_w, spatial_w, 14, v, [0.5, 1])
            inputs = rank(inputs, channel_w, spatial_w, 14, v,[0.5, 1])
            output = model(inputs)
            acc_num = topk_accuracy(output, target, 1)
            loss = criterion(output, target)
            val_acc_num += acc_num.item()
            val_loss += loss.item()
            val_steps += 1
        loop.set_description(f'Epoch:val [{i}/{len(testloader)}]')
        loop.set_postfix(loss=val_loss / val_steps, acc=val_acc_num / val_sample_num)
    if val_acc_num / val_sample_num > best_acc or best_acc == 0:
        best_acc = val_acc_num / val_sample_num
        torch.save(model.state_dict(), model_save_dir + "final_model.pth")
        torch.save(rank_moudle.state_dict(), model_save_dir + "rank.pth")
    return best_acc

def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Entromix Training")
    parser.add_argument(
        "--workers",
        default=2,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 2)",
    )
    parser.add_argument(
        "--epochs",
        default=90,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--batch-size",
        default=128,
        type=int,
        metavar="N",
        help="approximate bacth size",
    )
    parser.add_argument(
        "--lr",
        default=0.1,
        type=float,
        help="initial learning rate",
    )
    parser.add_argument(
        "--min-lr",
        default=0.001,
        type=float,
        help="Last learning rate",
    )
    parser.add_argument(
        "--evaluate",
        dest="evaluate",
        action="store_true",
        help="evaluate model on validation set",
    )
    parser.add_argument(
        "--model-save-dir",
        type=str,
        default="checkpoint",
        help="path to save model",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="./datas",
        help="Where data is be stored",
    )
    parser.add_argument(
        "--optim",
        type=str,
        default="SGD",
        help="Optimizer to use (Adam, SGD)",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device on which to run the code."
    )
    parser.add_argument(
        "--model", type=str, default="resnet", help="model to trainning."
    )
    parser.add_argument(
        "--cls", type=int, default="1000"
    )
    parser.add_argument(
        "--dataset", type=str, default="mnist"
    )

    return parser.parse_args()

def setEnv(args):
    cls = args.cls
    device = args.device
    if args.model == "resnet":
        model = models.resnet50(pretrained=False, num_classes=cls).to(device)
    elif args.model == "vgg":
        model = models.vgg16(pretrained=False, num_classes=cls).to(device)
    elif args.model == "vit":
        model = models.vit_b_16(num_classes=cls)
    elif args.model == "swin":
        model = models.swin_v2_s(num_classes=cls)
    model.to(device)
    rank_moudle = AttentionBlock(3, 3).to(device)
    dataset = args.dataset
    data_root = args.data_root
    if dataset == 'imagenet1k':
        # 实例化训练数据集
        train_dataset = torchvision.datasets.ImageFolder(data_root,
                                                         transform=torchvision.transforms.Compose([
                                                             torchvision.transforms.ToTensor(),
                                                             torchvision.transforms.Normalize(
                                                                 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                                         ]))
        # 实例化验证数据集
        val_dataset = torchvision.datasets.ImageFolder(data_root,
                                                       transform=torchvision.transforms.Compose([
                                                           transforms.RandomResizedCrop(224),
                                                           torchvision.transforms.ToTensor(),
                                                           torchvision.transforms.Normalize(
                                                               mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                                       ]))
    elif dataset == 'cifar10':
        val_dataset = torchvision.datasets.CIFAR10(data_root, train=False, download=True,
                                                   transform=torchvision.transforms.Compose([
                                                       transforms.Resize([224, 224]),
                                                       torchvision.transforms.ToTensor(),
                                                       torchvision.transforms.Normalize(
                                                           mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                                   ]))
        train_dataset = torchvision.datasets.CIFAR10(data_root, train=True, download=True,
                                                     transform=torchvision.transforms.Compose([
                                                         transforms.Resize([224, 224]),
                                                         torchvision.transforms.ToTensor(),
                                                         torchvision.transforms.Normalize(
                                                             mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                                     ]))
    elif dataset == 'mnist':
        val_dataset = torchvision.datasets.MNIST(data_root, train=False, download=True,
                                                   transform=torchvision.transforms.Compose([
                                                       transforms.Grayscale(num_output_channels=3),
                                                       transforms.Resize([224, 224]),
                                                       torchvision.transforms.ToTensor(),
                                                       torchvision.transforms.Normalize(
                                                           mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                                   ]))
        train_dataset = torchvision.datasets.MNIST(data_root, train=True, download=True,
                                                     transform=torchvision.transforms.Compose([
                                                         transforms.Grayscale(num_output_channels=3),
                                                         transforms.Resize([224, 224]),
                                                         torchvision.transforms.ToTensor(),
                                                             torchvision.transforms.Normalize(
                                                                 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                                     ]))
    pg = [p for p in model.parameters() if p.requires_grad]
    for p in rank_moudle.parameters():
        if p.requires_grad:
            pg.append(p)
    if args.optim == "Adam":
        optimizer = optim.Adam(pg, lr=args.lr)
    elif args.optim == "SGD":
        optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9)
    schedule = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    trainloader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              pin_memory=False,
                                              num_workers=args.workers)
    testloader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             pin_memory=False,
                                             num_workers=args.workers)
    criterion = nn.CrossEntropyLoss()
    return model, trainloader, testloader, criterion, schedule, rank_moudle, optimizer


if __name__ == '__main__':
    args = parse_args()
    model, trainloader, testloader, criterion, schedule, rank_moudle, optimizer = setEnv(args)
    best_acc = 0
    for i in range(args.epochs):
        train(trainloader, model, rank_moudle, args.device, optimizer, schedule, criterion)
        test(model, rank_moudle, testloader, best_acc, args.device, criterion, args.model_save_dir)
