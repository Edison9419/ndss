import numpy as np
from PIL import Image
import os
import time
import sys
from tqdm import *


def partial_derivative(img):
    img = np.pad(img, ((0, 1), (0, 1)), constant_values=0)
    h, w = img.shape
    df_gray = np.zeros([h - 1, w - 1])
    for i in range(h - 1):
        for j in range(w - 1):
            dx_gray = img[i, j + 1] - img[i, j]
            dy_gray = img[i + 1, j] - img[i, j]
            df_gray[i, j] = np.square(dx_gray) + np.square(dy_gray)
    return df_gray


def vfe(img: object, size: object, stride: object) -> object:
    image = []
    target = []
    num_h = (img.shape[0] - size) // stride + 1
    num_w = (img.shape[1] - size) // stride + 1
    for h in range(num_h):
        for w in range(num_w):
            img_crop = img[h * stride:h * stride + size, w * stride:w * stride + size]
            image.append(img_crop)
            crop = img_crop - np.mean(img_crop)
            crop = crop * crop
            target.append(crop / (stride * stride - 1))
    entropy = 0
    for crop in image:
        crop = partial_derivative(crop)
        entropy += np.sum(crop)
    entropy = entropy / len(image)
    return entropy, np.mean(target)


def vfe_shuffle(img: object, size: object, stride: object):
    image = []
    vfe = []
    num_h = (img.shape[0] - size) // stride + 1
    num_w = (img.shape[1] - size) // stride + 1
    for h in range(num_h):
        for w in range(num_w):
            img_crop = img[h * stride:h * stride + size, w * stride:w * stride + size]
            image.append(img_crop)
            entropy = np.sum(partial_derivative(img_crop))
            crop = img_crop - np.mean(img_crop)
            crop = crop * crop
            if np.sum(crop) == 0:
                vfe.append(0)
            else:
                vfe.append((entropy / np.sum(crop)) * (stride * stride) - 1)
    median = np.median(vfe)
    shuffled_crop = []
    for i, crop in enumerate(image):
        n = 1
        score = vfe[i]
        if score <= median:
            n = 2
        elif median < score:
            n = 2
        m = crop.shape[1]
        block_size = m // n
        blocks = []
        for i in range(0, m, block_size):
            for j in range(0, m, block_size):
                block = crop[i:i + block_size, j:j + block_size]
                blocks.append(block)
        np.random.shuffle(blocks)  
        shuffled_crop.append(np.vstack([np.hstack(blocks[i:i + n]) for i in range(0, len(blocks), n)]))
    shuffle_img = np.vstack([np.hstack(shuffled_crop[i:i + num_w]) for i in range(0, len(shuffled_crop), num_h)])
    return shuffle_img


def shuffle(img, size, stride):
    img[:, :, 0] = vfe_shuffle(img[:, :, 0], size, stride)
    img[:, :, 1] = vfe_shuffle(img[:, :, 1], size, stride)
    img[:, :, 2] = vfe_shuffle(img[:, :, 2], size, stride)
    return img

path = './datas/imagenet100/train/'
shuffle_range = [i + 10*int(sys.argv[1]) for i in range(10)]
for i, label in enumerate(os.listdir(path)):
    if i not in shuffle_range:
        print("skip:"+label)
        continue
    for img in tqdm(os.listdir(path+label+"")):
        try:
            image = Image.open(path + label + "/" + img)
        except:
            pass
        else:
            image = image.resize((112, 112))
            img_array = np.array(image)
            if len(img_array.shape) == 2:
                img_array = np.stack((img_array, img_array, img_array), axis=-1)
            img_array_shuffled = shuffle(img_array, 8, 8)
            save_path = "./datas/100_shuffle_112_4_4_2_4/train/" + label + "/"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            img_shuffled = Image.fromarray(np.uint8(img_array_shuffled))
            image = image.resize((224,224))
            img_shuffled.save(save_path+img)
