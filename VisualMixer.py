import random
import numpy as np
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

Settings = [[8, 8, 8, 8], [4, 8, 8, 8], [2, 4, 8, 8]]


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class AttentionBlock(nn.Module):
    def __init__(self, c1, c2):
        super(AttentionBlock, self).__init__()
        self.channel_attention = ChannelAttention(c1, c2)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        channel_w = self.channel_attention(x)
        out = channel_w * x
        spatial_w = self.spatial_attention(out)
        out = spatial_w * out
        return out, channel_w, spatial_w

def partial_derivative(img):
    img = F.pad(img, (0, 1, 0, 1), value=0)
    h, w = img.shape
    df_gray = torch.zeros((h - 1, w - 1))
    for i in range(h - 1):
        for j in range(w - 1):
            dx_gray = img[i, j + 1] - img[i, j]
            dy_gray = img[i + 1, j] - img[i, j]
            df_gray[i, j] = torch.square(dx_gray) + torch.square(dy_gray)
    return df_gray

@torch.no_grad()
def vfe(img: object, size: object, stride: object) -> object:
    image = []
    target = []
    num_h = (img.shape[1] - size) // stride + 1
    num_w = (img.shape[2] - size) // stride + 1
    for c in range(3):
        for h in range(num_h):
            for w in range(num_w):
                img_crop = img[c, h * stride:h * stride + size, w * stride:w * stride + size]
                image.append(img_crop)
                crop = img_crop - torch.mean(img_crop)
                crop = crop * crop
                target.append(crop / (stride * stride - 1))
    entropy = 0
    for crop in image:
        crop = partial_derivative(crop)
        entropy += torch.sum(crop)
    entropy = entropy / len(image)
    target = torch.cat(target)
    return entropy.item(), torch.mean(target)

def shuffle(imgs: Tensor, score: float, setting):
    n = 0
    if score <= 0.1:
        n = setting[0]
    elif 0.1 < score <= 0.5:
        n = setting[1]
    elif 0.5 < score <= 0.9:
        n = setting[2]
    elif 0.9 < score <= 1:
        n = setting[3]
    slices = torch.chunk(imgs, n, 1)
    chunks = []
    for slice in slices:
        temp = torch.chunk(slice, n, dim=2)
        for t in temp:
            chunks.append(t)
    random.shuffle(chunks)
    res = []
    for i in range(n):
        res.append(torch.cat(chunks[n * i:n * (i + 1)], dim=1))
    return torch.cat(res, dim=2)


def rank(imgs, channle_w, spatial_W, k, vfe, thresholds):
    assert len(imgs.shape) == 4
    with torch.no_grad():
        slices = torch.chunk(imgs, k, 2)
        chunks = []
        num = int(imgs.shape[2] / k)
        channle_w = channle_w.repeat(1, 1, k, k)
        pooling = torch.nn.AvgPool2d(kernel_size=num, stride=num)
        spatial_W = pooling(spatial_W)
        scores = (channle_w * spatial_W).detach()
        scores = torch.mean(scores, dim=0).view(-1)
        for slice in slices:
            temp = torch.chunk(slice, k, dim=3)
            for t in temp:
                chunks.append(t)
        max = torch.max(scores)
        min = torch.min(scores)
        if vfe >= thresholds[1]:
            setting = Settings[0]
        elif vfe < thresholds[0]:
            setting = Settings[2]
        else:
            setting = Settings[1]

        for i in range(imgs.shape[1]):
            for j in range(len(chunks)):
                score = (scores[i * k * k + j] - min) / (max - min).item()
                chunks[j][:, i, :, :] = shuffle(chunks[j][:, i, :, :], score, setting)
        res = []
        for i in range(k):
            tmp = []
            for j in range(k):
                tmp.append(chunks[i * k + j])
            res.append(torch.cat(tmp, dim=3))
        return torch.cat(res, dim=2)
