'''
Date: 2022-03-02 21:24:42
Author: ChHanXiao
Github: https://github.com/ChHanXiao
LastEditors: ChHanXiao
LastEditTime: 2022-03-02 21:26:58
FilePath: /license-plate-recoginition/datasets/prepare_ccpd.py
'''
import os
import cv2
import numpy as np
import time
ccpd_root_path_dict = {
    'ccpd2020': '/media/hanxiao/data/data/LPR/CCPD2020',
    'ccpd2019': '/media/hanxiao/data/data/LPR/CCPD2019'
}

ccpd_crop_path_dict = {
    'ccpd2020': '/media/hanxiao/data/data/LPR/CCPD2020/crop',
    'ccpd2019': '/media/hanxiao/data/data/LPR/CCPD2019/crop'
}

ccpd_splits_path_dict = {
    'ccpd2020': '/media/hanxiao/data/data/LPR/CCPD2020/splits',
    'ccpd2019': '/media/hanxiao/data/data/LPR/CCPD2019/splits'
}
provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

# CCPD2020 train/
# 0111328125-90_264-229&500_419&559-419&559_234&557_229&500_413&502-1_0_33_1_32_26_25_25-121-148.jpg
# 沪AF98821---->1_0_5_33_32_32_26_25
# 0111328125-90_264-229&500_419&559-419&559_234&557_229&500_413&502-1_0_5_33_32_32_26_25-121-148.jpg

# 0181467013889-88_104-148&444_374&525-374&513_161&525_148&455_356&444-0_0_24_26_30_27_30_24-135-18.jpg
# 皖AD26360---->0_0_3_26_30_27_30_24
# 0181467013889-88_104-148&444_374&525-374&513_161&525_148&455_356&444-0_0_3_26_30_27_30_24-135-18.jpg

# check is wrong
def check(label):
    # green LP :label[2] label[-1] should be 'D' or 'F'
    # WJ LP : label[0]='W' label[1]='J'
    if label[2] != 3 and label[2] != 5 \
            and label[-1] != 3 and label[-1] != 5 \
            and label[0] != 20:
        print("Error label, Please check!")
        return False
    else:
        return True

def get_crop(dataset='ccpd2019'):
    dignum = 5 if dataset =='ccpd2019' else 6
    # lp_type = '04' if dataset == 'ccpd2019' else '05'
    lp_color = 'blue' if dataset == 'ccpd2019' else 'green+s'
    ccpd_root_path = ccpd_root_path_dict[dataset]
    ccpd_crop_path = ccpd_crop_path_dict[dataset]
    ccpd_splits_path = ccpd_splits_path_dict[dataset]
    for train_set in ['train','test','val']:
        with open(os.path.join(ccpd_splits_path, train_set)+'.txt') as f:
            imgs_name = f.readlines()
            total_len = len(imgs_name)
            for idx, img_name in enumerate(imgs_name):
                if idx % 500 == 0:
                    print('process:'+str(idx)+',total:'+str(total_len))
                lbl4point = img_name.split('/')[-1].rsplit('.', 1)[0].split('-')[-4]
                img_src = cv2.imread(os.path.join(ccpd_root_path, img_name.strip()))
                lprloca = [[int(eel) for eel in el.split('&')] for el in lbl4point.split('_')]
                p0 = np.float32([lprloca[2],lprloca[1],lprloca[3],lprloca[0]])
                p1 = np.float32([(0,0),(0,80),(240,0),(240,80)])
                transform_mat=cv2.getPerspectiveTransform(p0,p1)
                lic = cv2.warpPerspective(img_src,transform_mat,(240,80))
                save_path = os.path.join(ccpd_crop_path, train_set)
                lprnum = img_name.split('/')[-1].rsplit('.', 1)[0].split('-')[4].split('_')
                lprnum = [int(eel) for eel in lprnum]
                if len(lprnum)==8:
                    if not check(lprnum):
                        save_path = os.path.join(ccpd_crop_path, train_set+'check')

                candidates = [provinces, alphabets] + [ads] * dignum
                label = ''.join(catlist[lprnum_] for lprnum_, catlist in zip(lprnum,candidates))

                ticks = str(int(round(time.time()*1000000)))+'.png'
                timemark = time.strftime("%Y%m%d", time.localtime()) 

                newname = timemark+'-'+lp_color+'-'+label+'-'+ticks
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                save_filename = os.path.join(save_path, newname)
                cv2.imwrite(save_filename,lic)


if __name__ == '__main__':
    dataset = 'ccpd2019'
    # LPRNet rec crop small img
    get_crop(dataset)







