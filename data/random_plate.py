import random

import numpy as np

from .chinese import *


class Draw_cfg:
    def __init__(self, cfg):
        self.cfg = cfg
        self._draw = [
            black_plate.Draw(),
            blue_plate.Draw(),
            yellow_plate.Draw(),
            green_plate.Draw(),
            white_plate.Draw(),
            farm_plate.Draw(),
            airport_plate.Draw()
        ]
        self._provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新"]
        self._alphabets = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
        self._ads = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        self._check = ["D", "F"]
        self._checknum = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        self._emergency = ["X", "S"] 
    def __call__(self):
        p=self.cfg.p
        draw = np.random.choice(self._draw, p=p)

        if type(draw) == green_plate.Draw:
            green_p = self.cfg.green_plate.p
            green_t = np.random.choice(Green_Type, p=green_p)
            candidates = [self._provinces, self._alphabets]
            # 渐变绿
            if green_t == Green_Type.SMALL:
                candidates += [self._check, ]
                candidates += [self._ads, ]
                candidates += [self._checknum] * 4
                # 调试 1正常生成、2连续字符、3易错字符、4全省份
                label_type=3 
                # label_type = random.choice([2,3])
                if label_type==1:
                    label = "".join([random.choice(c) for c in candidates])
                elif label_type == 2:
                    tmp = [random.choice(c) for c in candidates[:4]]
                    tmp1 = random.choice(self._checknum)
                    tmp2 = tmp+[tmp1]*4
                    label = "".join(tmp2)
                elif label_type == 3:
                    _checknum_tmp = ["0", "Q", "5", "6", "2", "9", "8"]
                    tmp = [random.choice(c) for c in candidates[:4]]
                    tmp11 = [random.choice(c) for c in [_checknum_tmp]*4]
                    tmp22 = tmp+tmp11
                    label = "".join(tmp22)
                elif label_type == 4:
                    candidates = [self._provinces]*8
                    tmp = [random.choice(c) for c in candidates]
                    label = "".join(tmp)
                return draw(label, green_t), label, 'green+s'
            # 黄绿双色
            elif green_t == Green_Type.LARGE:
                candidates += [self._checknum] * 5
                candidates += [self._check, ]
                label_type=3   # 调试 1正常生成、2连续字符、3易错字符、4全省份

                if label_type==1:
                    label = "".join([random.choice(c) for c in candidates])
                if label_type == 2:
                    tmp = [random.choice(c) for c in candidates[:2]]
                    tmp1 = random.choice(self._checknum)
                    tmp2 = tmp + [tmp1] * 5
                    tmp3 = tmp2 + [random.choice(self._check)]
                    label = "".join(tmp3)
                if label_type == 3:
                    tmp = [random.choice(c) for c in candidates[:2]]
                    _checknum_tmp = ["0", "Q", "5", "6", "2", "9", "8"]
                    tmp11 = [random.choice(c) for c in [_checknum_tmp]*5]
                    tmp22 = tmp+tmp11
                    tmp33 = tmp22 + [random.choice(self._check)]
                    label = "".join(tmp33)
                if label_type == 4:
                    candidates = [self._provinces]*8
                    tmp = [random.choice(c) for c in candidates]
                    label = "".join(tmp)
                return draw(label, green_t), label, 'green+b'

        elif type(draw) == blue_plate.Draw:
            candidates = [self._provinces, self._alphabets]
            candidates += [self._ads] * 5
            label_type = 1  # 调试 1正常生成、2连续字符、3易错字符、4全省份
            if label_type == 1:
                label = "".join([random.choice(c) for c in candidates])
            elif label_type == 4:
                candidates = [self._provinces]*7
                tmp = [random.choice(c) for c in candidates]
                label = "".join(tmp)
            return draw(label), label, 'blue'

        elif type(draw) == yellow_plate.Draw:
            yellow_p = self.cfg.yellow_plate.p
            yellow_t = np.random.choice(Yellow_Type, p=yellow_p)
            if yellow_t == Yellow_Type.SINGLE:
                candidates = [self._provinces, self._alphabets]
                label_type = 1  # 调试 1正常生成、2连续字符、3易错字符、4全省份
                if random.random() < 0.1:
                    candidates += [self._ads] * 4
                    candidates += [["学"]]
                else:
                    candidates += [self._ads] * 5
                if label_type == 1:
                    label = "".join([random.choice(c) for c in candidates])
                elif label_type == 4:
                    candidates = [self._provinces]*7
                    tmp = [random.choice(c) for c in candidates]
                    label = "".join(tmp)
                return draw(label, bg_type=yellow_t), label, 'yellow+s'
            elif yellow_t == Yellow_Type.MULTI:
                candidates = [self._provinces, self._alphabets]
                if random.random() < 0.1:
                    candidates += [self._ads] * 4
                    candidates += [["挂"]]
                else:
                    candidates += [self._ads] * 5
                label = "".join([random.choice(c) for c in candidates])
                return draw(label, bg_type=yellow_t), label, 'yellow+m'
            else:
                raise NotImplementedError

        elif type(draw) == black_plate.Draw:
            black_p = self.cfg.black_plate.p
            black_t = np.random.choice(Black_Type, p=black_p)
            if black_t == Black_Type.COMMON:
                if random.random()<0.5:
                    candidates = [["粤"], self._alphabets]
                    candidates += [self._ads] * 4
                    candidates += [["港", "澳"]]
                    label = "".join([random.choice(c) for c in candidates])
                    return draw(label,bg_type=black_t), label, 'black'
                else:
                    candidates = [self._provinces, self._alphabets]
                    candidates += [self._ads] * 5
                    label = "".join([random.choice(c) for c in candidates])
                    return draw(label,bg_type=black_t), label, 'black'
            elif black_t == Black_Type.LIN:
                candidates = [self._provinces, self._alphabets]
                candidates += [self._ads] * 4
                candidates += [["领"]]
                label = "".join([random.choice(c) for c in candidates])
                return draw(label,bg_type=black_t), label, 'black'
            elif black_t == Black_Type.SHI:
                candidates = [["使"]]
                candidates += [self._checknum] * 6
                label = "".join([random.choice(c) for c in candidates])
                return draw(label,bg_type=black_t), label, 'black'
            else:
                raise NotImplementedError
       
        elif type(draw) == white_plate.Draw:
            white_p = self.cfg.white_plate.p
            front_p= self.cfg.white_plate.front
            white_t = np.random.choice(White_Type, p=white_p)
            front_ = True if random.random()>front_p else False
            if white_t == White_Type.JINGCHA:  #单层警牌7位：省份+地区+字母数组+数字*3+"警"
                candidates = [self._provinces, self._alphabets]
                candidates += [self._ads]
                candidates += [self._checknum] * 3
                candidates += [["警"]]
                label = "".join([random.choice(c) for c in candidates])
                return draw(label, bg_type=white_t,front_=front_), label, 'white+jc'
            elif white_t == White_Type.JUNDUI:  #单层军牌7位：字母*2+数字*5
                candidates = [self._alphabets]*2
                candidates += [self._checknum] * 5
                label = "".join([random.choice(c) for c in candidates])
                return draw(label, bg_type=white_t,front_=front_), label, 'white+jd'
            elif white_t == White_Type.WUJIN:  #单层武警牌8位："WJ"+字母+数字*5
                candidates = [["W"],["J"]]
                candidates += [self._alphabets]
                candidates += [self._checknum] * 5
                label = "".join([random.choice(c) for c in candidates])
                return draw(label, bg_type=white_t,front_=front_), label, 'white+wj'
            elif white_t == White_Type.YINGJI: #应急车牌8位：省份+类型+数字*4+"应急"
                candidates = [self._provinces] 
                candidates += [self._emergency]
                candidates += [self._checknum] * 4
                candidates += [["应"], ["急"]]
                label = "".join([random.choice(c) for c in candidates])
                return draw(label, bg_type=white_t,front_=front_), label, 'white+yj'
            elif white_t == White_Type.JUNDUI_M: #双层军牌7位：字母+字母+数字*5
                candidates = [self._alphabets] * 2
                candidates += [self._checknum] * 5
                label = "".join([random.choice(c) for c in candidates])
                return draw(label, bg_type=white_t,front_=front_), label, 'white+jdm'
            elif white_t == White_Type.WUJIN_M: #双层武警牌8位："WJ"+字母+数字*5
                candidates = [["W"],["J"]]
                candidates += [self._alphabets] 
                candidates += [self._checknum] * 5
                label = "".join([random.choice(c) for c in candidates])
                return draw(label, bg_type=white_t,front_=front_), label, 'white+wjm'
            else:
                raise NotImplementedError

        elif type(draw) == farm_plate.Draw:
            farm_p = self.cfg.farm_plate.p
            farm_t = np.random.choice(Farm_Type, p=farm_p)
            candidates = [self._provinces]
            if farm_t == Farm_Type.XUE:
                candidates += [self._checknum] * 6
                candidates += [["学"]]
                label = "".join([random.choice(c) for c in candidates])
                return draw(label, bg_type=farm_t), label, 'farm+m'
            elif farm_t == Farm_Type.COMMON:
                candidates += [self._checknum] * 7
                label = "".join([random.choice(c) for c in candidates])
                return draw(label, bg_type=farm_t), label, 'farm+m'
            else:
                raise NotImplementedError
        
        elif type(draw) == airport_plate.Draw:
            airport_p = self.cfg.airport_plate.p
            airport_t = np.random.choice(Airport_Type, p=airport_p)
            if airport_t == Airport_Type.COMMON:
                candidates = [["民"], ["航"]]
                candidates += [self._ads] * 5
                label = "".join([random.choice(c) for c in candidates])
                return draw(label, bg_type=airport_t), label, 'airport'
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError


if __name__ == "__main__":
    import argparse
    import math

    import matplotlib.pyplot as plt

    from data.config_gen import cfg

    parser = argparse.ArgumentParser(description="Generate a green plate.")
    parser.add_argument("--num", help="set the number of plates (default: 9)", type=int, default=20)
    args = parser.parse_args()
    draw = Draw_cfg(cfg)
    rows = math.ceil(args.num / 5)
    cols = min(args.num, 5)
    i=10000
    while i>0:
        plate, label, _ = draw()
        print(label)
        i=i-1
    for i in range(args.num):
        plate, label, _ = draw()
        print(label)
        plt.subplot(rows, cols, i + 1)
        plt.imshow(plate)
        plt.axis("off")
    plt.show()
