import os
import random
import numpy as np
import cv2
import argparse
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
from math import *
import albumentations as A


aug = A.Compose([
    A.OneOf([  #雨雾日光阴影
        A.RandomRain(drop_length=5, blur_value=4, brightness_coefficient=0.8, p=0.5),
        A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.5, alpha_coef=0.05, p=0.2),
        A.RandomShadow(shadow_roi=(0.1,0.,0.9,0.6), shadow_dimension=4, p=0.2),
        A.RandomSunFlare(flare_roi=(0.1,0.1,1.,1.), src_radius=100, p=0.2),
        # A.RandomSnow(snow_point_lower=0.1, snow_point_upper=0.2, brightness_coeff=2, p=0.1)
        ], p=1),
    A.MotionBlur(p=0.3),  # 动态模糊
    A.GaussianBlur(blur_limit=(3, 15), sigma_limit=0.4, p=1),  #高斯模糊
    A.OneOf([ #模糊
        A.MedianBlur(blur_limit=7, p=0.8),  #中值模糊blur_limit为奇
        A.Blur(blur_limit=9, p=0.8),
        A.GlassBlur(sigma=0.7, max_delta=3, p=0.2) #棱镜模糊
        ], p=0.5),
    A.OneOf([  #噪声
        A.ISONoise(color_shift=(0.04, 0.08), intensity=(0.2, 0.6), p=0.9),
        A.IAAAdditiveGaussianNoise(scale=(0.02 * 255, 0.07 * 255), p=0.9),
        A.MultiplicativeNoise(multiplier=(0.8, 1.2), p=0.9),
        A.GaussNoise(var_limit=(40.0, 100.0), p=1),
        ], p=1),
    A.CoarseDropout(max_holes=4, max_height=15, max_width=15, p=0.2),  #挖小洞
    #非刚体变换方法
    A.OpticalDistortion(distort_limit=0.2, shift_limit=0.2, border_mode=cv2.BORDER_CONSTANT, p=1),
    A.IAAPiecewiseAffine(scale=(0.02, 0.03), nb_cols=3, nb_rows=2, p=1), #网格仿射变换
    A.RandomBrightnessContrast(brightness_limit=0.4, p=0.7),  # 随机光照和对比度
    A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, always_apply=False, p=0.5),
    A.OneOf([
        A.RandomGamma(gamma_limit=(60, 120), p=0.4), #伽马变换
        A.CLAHE(clip_limit=2, p=0.4),  #自适应直方图均衡
        A.IAASharpen(p=0.4),  #锐化
        A.IAAEmboss(p=0.4),  #浮雕
        ], p=0.4),
    ], p=1)


def Brightness(img, delta):
    img = np.uint8(np.clip(img*(1+random.uniform(-delta, delta)), 0, 255))
    return img

def gamma_transform(img):
    gamma = 1 + (random.random()-0.5)
    gamma_table = [np.power(x/255.0, gamma)*255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    img = cv2.LUT(img, gamma_table)
    return img

def AddSmudginess(img, Smu_set):
    bg_img_path = Smu_set[r(len(Smu_set))]
    env = cv2.imread(bg_img_path)
    img_h, img_w = img.shape[:2]
    rows = r(env.shape[0] - img_h)
    cols = r(env.shape[1] - img_w)
    adder = env[rows:rows + img_h, cols:cols + img_w]
    adder = cv2.resize(adder, (img_w, img_h))
    adder = cv2.bitwise_not(adder)
    val = random.random() * 0.3
    img = cv2.addWeighted(img, 1 - val, adder, val, 0.0)
    return img

def rot(img,angel,shape,max_angel):
    """ 使图像轻微的畸变
        img 输入图像
        factor 畸变的参数
        size 为图片的目标尺寸
    """
    size_o = [shape[1],shape[0]]
    # size = (shape[1]+ int(shape[0]*cos((float(max_angel )/180) * 3.14)),shape[0])
    size = (shape[1]+ int(shape[0]*sin((float(max_angel )/180) * 3.14)),shape[0])
    interval = abs(int(sin((float(angel) /180) * 3.14)* shape[0]))

    pts1 = np.float32([[0,0],[0,size_o[1]],[size_o[0],0],[size_o[0],size_o[1]]])
    if(angel>0):
        pts2 = np.float32([[interval,0],[0,size[1]],[size[0],0],[size[0]-interval,size_o[1]]])
    else:
        pts2 = np.float32([[0,0],[interval,size[1]],[size[0]-interval,0],[size[0],size_o[1]]])

    M  = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(img,M,size)
    imgh,imgw = img.shape[0],img.shape[1]
    rot_mask = np.ones([imgh,imgw], dtype=np.uint8)
    rot_mask = cv2.warpPerspective(rot_mask,M,size)
    return dst, rot_mask

def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """
    height, width = mat.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape
    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)
    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0]) 
    abs_sin = abs(rotation_mat[0,1])
    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)
    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]
    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))

    rot_mask = np.ones([height,width], dtype=np.uint8)
    rot_mask = cv2.warpAffine(rot_mask, rotation_mat, (bound_w, bound_h))
    return rotated_mat, rot_mask

def rotRandrom(img, factor, size):
    shape = size
    pts1 = np.float32([[0, 0], [0, shape[0]], [shape[1], 0], [shape[1], shape[0]]])
    pts2 = np.float32([[r(factor), r(factor)],
                        [r(factor), shape[0] - r(factor)],
                        [shape[1] - r(factor), r(factor)],
                        [shape[1] - r(factor), shape[0] - r(factor)]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, size)
    return dst

def tfactor(img):
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    hsv[:,:,0] = hsv[:,:,0]*(0.8+ np.random.random()*0.2)
    hsv[:,:,1] = hsv[:,:,1]*(0.3+ np.random.random()*0.7)
    hsv[:,:,2] = hsv[:,:,2]*(0.2+ np.random.random()*0.8)
    img = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    return img

def random_envirment(img,data_set):
    index=r(len(data_set))
    env = cv2.imread(data_set[index])
    env = cv2.resize(env,(img.shape[1],img.shape[0]))
    val = random.random() * 0.4
    img = cv2.addWeighted(img, 1 - val, env, val, 0.0)
    return img

def random_xy_shift(img, size, factor, factor_base=5):
    shape = size
    pts1 = np.float32([[0, 0], [0, shape[0]], [shape[1], 0], [shape[1], shape[0]]])
    pts2 = np.float32([[r(factor)+factor_base, r(factor)+factor_base],
                        [r(factor), shape[0] - r(factor)],
                        [shape[1] - r(factor), r(factor)],
                        [shape[1] - r(factor)-factor_base, shape[0] - r(factor)-factor_base]])
    M = cv2.getPerspectiveTransform(pts2, pts1)
    dst = cv2.warpPerspective(img, M, size)
    return dst

def random_crop(img, factor, factor_base=3):
    left_shift = int(factor_base+random.random()*factor)
    right_shift = int(factor_base+random.random()*factor)
    top_shift = int(factor_base+random.random()*factor)
    down_shift = int(factor_base+random.random()*factor)
    img = img[top_shift:-down_shift, left_shift:-right_shift, :]
    return img

def random_scene_small(img, data_set, rot_mask=None, factor=10, factor_base=5):
    '''将车牌放入自然场景图像中，并返回该图像'''
    bg_img_path = data_set[r(len(data_set))]
    env = cv2.imread(bg_img_path)
    if env is None:
        print(bg_img_path, 'is not a good file')
        return None, None

    # # 原图随机crop
    # if random.random() < p_shift:
    #     left_shift = int(4 + random.random()*7)
    #     right_shift = int(4 + random.random()*7)
    #     top_shift = int(6 + random.random()*9)
    #     down_shift = int(6 + random.random()*9)
    #     img = img[top_shift:-down_shift, left_shift:-right_shift, :]
    #     rot_mask = rot_mask[top_shift:-down_shift, left_shift:-right_shift]
    #     return img

    new_height = int(img.shape[0] + factor_base+ np.random.random()*factor*0.8)
    new_width = int(img.shape[1] + factor_base+ np.random.random()*factor)
    env = cv2.resize(env, (new_width, new_height))
    x_shift_max = (env.shape[1] - img.shape[1]-1)
    y_shift_max = (env.shape[0] - img.shape[0]-1)
    x_shift = int(np.random.uniform(0,x_shift_max))
    y_shift = int(np.random.uniform(0,y_shift_max))

    ret, rot_mask = cv2.threshold(rot_mask, 0, 255, cv2.THRESH_BINARY)
    frontimg = cv2.bitwise_and(img, img, mask=rot_mask)
    img_roi = env[y_shift:y_shift+img.shape[0], x_shift:x_shift+img.shape[1], :]
    notmask = cv2.bitwise_not(rot_mask)
    backimg = cv2.bitwise_and(img_roi, img_roi, mask=notmask)
    addpic = cv2.add(backimg, frontimg)
    env[y_shift:y_shift + img.shape[0], x_shift:x_shift + img.shape[1], :]=addpic
    return env

def random_scene(img, data_set):
    '''将车牌放入自然场景图像中，并返回该图像和位置信息'''
    bg_img_path = data_set[r(len(data_set))]
    # print bg_img_path
    env = cv2.imread(bg_img_path)
    if env is None:
        print(bg_img_path, 'is not a good file')
        return None, None
    # print env.shape, img.shape
    # 如果背景图片小于（65，21）则不使用
    if env.shape[1] <= 65 or env.shape[0] <= 21:
        print(env.shape)
        return None, None
    # 车牌宽高比变化范围是(1.5, 4.0)
    new_height = img.shape[0] * (0.5 + np.random.random()) # 0.5 -- 1.5
    new_width = img.shape[1] * (0.5 + np.random.random()) # 0.5 -- 1.5
    scale = 0.3 + np.random.random() * 2.5
    new_width = int(new_width * scale + 0.5)
    new_height = int(new_height * scale + 0.5)
    img = cv2.resize(img, (new_width, new_height))
    if env.shape[1] <= img.shape[1] or env.shape[0] <= img.shape[0]:
        print(env.shape, '---', img.shape)
        return None, None
    x = r(env.shape[1] - img.shape[1])
    y = r(env.shape[0] - img.shape[0])
    bak = (img==0)
    bak = bak.astype(np.uint8)*255
    inv = cv2.bitwise_and(bak, env[y:y+new_height, x:x+new_width, :])
    img = cv2.bitwise_or(inv, img)
    env[y:y+new_height, x:x+new_width, :] = img[:,:,:]

    return env, (x, y, x + img.shape[1], y + img.shape[0])

def GenCh(f,val):
    img=Image.new("RGB", (45,70),(255,255,255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 3),val,(0,0,0),font=f)
    img = img.resize((23,70))
    A = np.array(img)
    return A

def GenCh1(f,val):
    img=Image.new("RGB", (23,70),(255,255,255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 2),val,(0,0,0),font=f)
    # draw.text((0, 2), val.decode('utf-8'), (0, 0, 0), font=f)
    A = np.array(img)
    return A

def AddGauss(img, factor=6, factor_base=2):
    level = int(factor_base + np.random.random()*factor)
    k_size = (level * 2 + 1, level * 2 + 1)
    return cv2.blur(img, k_size)

def r(val):
    return int(np.random.random() * val)

def AddNoiseSingleChannel(single):
    diff = 255-single.max()
    noise = np.random.normal(0,2+r(4),single.shape)
    noise = (noise - noise.min())/(noise.max()-noise.min())
    noise= diff*noise
    noise= noise.astype(np.uint8)
    dst = single + noise
    return dst

def addNoise(img,sdev = 0.5,avg=10):
    img[:,:,0] = AddNoiseSingleChannel(img[:,:,0])
    img[:,:,1] = AddNoiseSingleChannel(img[:,:,1])
    img[:,:,2] = AddNoiseSingleChannel(img[:,:,2])
    return img
