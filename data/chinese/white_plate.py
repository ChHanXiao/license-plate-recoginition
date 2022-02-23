import os
import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import random
from enum import Enum
class White_Type(Enum):
    COMMON = 0      # 7位
    JINGCHA = 1     # 7位
    JUNDUI = 2      # 7位
    JUNDUI_M = 3    # 7位
    WUJIN = 4       # 8位
    WUJIN_M = 5     # 8位
    YINGJI = 6      # 8位


def load_font():
    return {
        "京": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_京.jpg")),
        "津": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_津.jpg")),
        "冀": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_冀.jpg")),
        "晋": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_晋.jpg")),
        "蒙": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_蒙.jpg")),
        "辽": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_辽.jpg")),
        "吉": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_吉.jpg")),
        "黑": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_黑.jpg")),
        "沪": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_沪.jpg")),
        "苏": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_苏.jpg")),
        "浙": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_浙.jpg")),
        "皖": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_皖.jpg")),
        "闽": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_闽.jpg")),
        "赣": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_赣.jpg")),
        "鲁": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_鲁.jpg")),
        "豫": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_豫.jpg")),
        "鄂": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_鄂.jpg")),
        "湘": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_湘.jpg")),
        "粤": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_粤.jpg")),
        "桂": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_桂.jpg")),
        "琼": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_琼.jpg")),
        "渝": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_渝.jpg")),
        "川": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_川.jpg")),
        "贵": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_贵.jpg")),
        "云": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_云.jpg")),
        "藏": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_藏.jpg")),
        "陕": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_陕.jpg")),
        "甘": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_甘.jpg")),
        "青": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_青.jpg")),
        "宁": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_宁.jpg")),
        "新": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_新.jpg")),
		# "澳": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_澳.jpg")),
		# "港": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_港.jpg")),
        # "学": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_学.jpg")),
        "警": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_警.jpg")),
        # "使": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_使.jpg")),
        # "领": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_领.jpg")),
        "A": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_A.jpg")),
        "B": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_B.jpg")),
        "C": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_C.jpg")),
        "D": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_D.jpg")),
        "E": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_E.jpg")),
        "F": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_F.jpg")),
        "G": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_G.jpg")),
        "H": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_H.jpg")),
        "J": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_J.jpg")),
        "K": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_K.jpg")),
        "L": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_L.jpg")),
        "M": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_M.jpg")),
        "N": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_N.jpg")),
        "P": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_P.jpg")),
        "Q": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_Q.jpg")),
        "R": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_R.jpg")),
        "S": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_S.jpg")),
        "T": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_T.jpg")),
        "U": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_U.jpg")),
        "V": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_V.jpg")),
        "W": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_W.jpg")),
        "X": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_X.jpg")),
        "Y": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_Y.jpg")),
        "Z": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_Z.jpg")),
		"0": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_0.jpg")),
        "1": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_1.jpg")),
        "2": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_2.jpg")),
        "3": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_3.jpg")),
        "4": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_4.jpg")),
        "5": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_5.jpg")),
        "6": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_6.jpg")),
        "7": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_7.jpg")),
        "8": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_8.jpg")),
        "9": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_9.jpg"))
    }
def load_font_up():
    return {
        "京": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_京.jpg")),
        "津": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_津.jpg")),
        "冀": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_冀.jpg")),
        "晋": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_晋.jpg")),
        "蒙": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_蒙.jpg")),
        "辽": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_辽.jpg")),
        "吉": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_吉.jpg")),
        "黑": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_黑.jpg")),
        "沪": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_沪.jpg")),
        "苏": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_苏.jpg")),
        "浙": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_浙.jpg")),
        "皖": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_皖.jpg")),
        "闽": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_闽.jpg")),
        "赣": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_赣.jpg")),
        "鲁": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_鲁.jpg")),
        "豫": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_豫.jpg")),
        "鄂": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_鄂.jpg")),
        "湘": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_湘.jpg")),
        "粤": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_粤.jpg")),
        "桂": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_桂.jpg")),
        "琼": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_琼.jpg")),
        "渝": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_渝.jpg")),
        "川": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_川.jpg")),
        "贵": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_贵.jpg")),
        "云": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_云.jpg")),
        "藏": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_藏.jpg")),
        "陕": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_陕.jpg")),
        "甘": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_甘.jpg")),
        "青": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_青.jpg")),
        "宁": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_宁.jpg")),
        "新": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_新.jpg")),
        "A": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_up_A.jpg")),
        "B": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_up_B.jpg")),
        "C": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_up_C.jpg")),
        "D": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_up_D.jpg")),
        "E": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_up_E.jpg")),
        "F": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_up_F.jpg")),
        "G": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_up_G.jpg")),
        "H": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_up_H.jpg")),
        "J": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_up_J.jpg")),
        "K": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_up_K.jpg")),
        "L": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_up_L.jpg")),
        "M": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_up_M.jpg")),
        "N": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_up_N.jpg")),
        "P": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_up_P.jpg")),
        "Q": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_up_Q.jpg")),
        "R": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_up_R.jpg")),
        "S": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_up_S.jpg")),
        "T": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_up_T.jpg")),
        "U": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_up_U.jpg")),
        "V": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_up_V.jpg")),
        "W": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_up_W.jpg")),
        "X": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_up_X.jpg")),
        "Y": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_up_Y.jpg")),
        "Z": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_up_Z.jpg")),
    }

def load_font_down():
    return {
        "挂": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_挂.jpg")),
        "A": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_down_A.jpg")),
        "B": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_down_B.jpg")),
        "C": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_down_C.jpg")),
        "D": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_down_D.jpg")),
        "E": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_down_E.jpg")),
        "F": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_down_F.jpg")),
        "G": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_down_G.jpg")),
        "H": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_down_H.jpg")),
        "J": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_down_J.jpg")),
        "K": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_down_K.jpg")),
        "L": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_down_L.jpg")),
        "M": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_down_M.jpg")),
        "N": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_down_N.jpg")),
        "P": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_down_P.jpg")),
        "Q": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_down_Q.jpg")),
        "R": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_down_R.jpg")),
        "S": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_down_S.jpg")),
        "T": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_down_T.jpg")),
        "U": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_down_U.jpg")),
        "V": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_down_V.jpg")),
        "W": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_down_W.jpg")),
        "X": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_down_X.jpg")),
        "Y": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_down_Y.jpg")),
        "Z": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_down_Z.jpg")),
        "0": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_0.jpg")),
        "1": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_1.jpg")),
        "2": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_2.jpg")),
        "3": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_3.jpg")),
        "4": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_4.jpg")),
        "5": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_5.jpg")),
        "6": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_6.jpg")),
        "7": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_7.jpg")),
        "8": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_8.jpg")),
        "9": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/220_9.jpg"))
    }
class Draw:
    def __init__(self):
        self.lplen = [7,8]
        self._font = load_font()
        self._font_up = load_font_up()
        self._font_down = load_font_down()

        self._bg_common = [
            cv2.resize(cv2.imread(os.path.join(os.path.dirname(__file__), f"res/white/white_{imgnum}.png")), (440, 140)) for imgnum in np.arange(2)
            ]
        self._bg_jingcha = [
            cv2.resize(cv2.imread(os.path.join(os.path.dirname(__file__), f"res/white/white_jc_{imgnum}.png")), (440, 140)) for imgnum in np.arange(2)
            ]
        self._bg_jundui = [
            cv2.resize(cv2.imread(os.path.join(os.path.dirname(__file__), f"res/white/white_jd_{imgnum}.png")), (490, 140)) for imgnum in np.arange(1)
            ]
        self._bg_jundui_m = [
            cv2.resize(cv2.imread(os.path.join(os.path.dirname(__file__), f"res/white/white_jd_m_{imgnum}.png")), (495, 240)) for imgnum in np.arange(1)
            ]
        self._bgs_wujing = [
            cv2.resize(cv2.imread(os.path.join(os.path.dirname(__file__), f"res/white/white_wj_{imgnum}.png")), (480, 140)) for imgnum in np.arange(1)
            ]
        self._bgs_wujing_m = [
            cv2.resize(cv2.imread(os.path.join(os.path.dirname(__file__), f"res/white/white_wj_m_{imgnum}.png")), (495, 240)) for imgnum in np.arange(1)
            ]
        self._bgs_yingji = [
            cv2.resize(cv2.imread(os.path.join(os.path.dirname(__file__), f"res/white/white_yj_{imgnum}.png")), (480, 140)) for imgnum in np.arange(1)
            ]

    def __call__(self, plate, bg_type=White_Type.COMMON, front_=False):
        if len(plate) not in self.lplen:
            print(f"ERROR: plate length:{plate},bg_type:{bg_type}")
            return None
        if (len(plate) == 7):
            if bg_type not in [White_Type.COMMON, White_Type.JINGCHA, White_Type.JUNDUI, White_Type.JUNDUI_M ]:
                print(f"ERROR: plate length:{plate},bg_type:{bg_type}")
                return None
        if (len(plate) == 8):
            if bg_type not in [White_Type.WUJIN, White_Type.WUJIN_M, White_Type.JUNDUI, White_Type.YINGJI ]:
                print(f"ERROR: plate length:{plate},bg_type:{bg_type}")
                return None        
        bg = self._draw_bg(bg_type)
        fg = self._draw_fg(plate, bg_type,p_front=front_)
        return cv2.cvtColor(cv2.bitwise_and(fg, bg), cv2.COLOR_BGR2RGB)

    def _draw_char(self, ch,):
            return cv2.resize(self._font[ch], (45, 90))
    def _draw_char_jd(self, ch,):
            return cv2.resize(self._font[ch], (50, 84))
    def _draw_char_yj(self, ch,):
            return cv2.resize(self._font[ch], (43, 90))
    def _draw_char_multi(self, ch, upper=True):
        if upper:
            return cv2.resize(self._font[ch], (100,70))
        else:
            return cv2.resize(self._font[ch], (73, 100))
    def _draw_char_wj(self, ch,):
            return cv2.resize(self._font[ch], (56, 70))
    def _draw_char_wj_multi(self, ch, upper=True):
        if upper:
            return cv2.resize(self._font[ch], (73, 70))
        else:
            return cv2.resize(self._font[ch], (73, 100))
    
    def _draw_fg(self, plate, bg_type, p_front):
        if bg_type==White_Type.COMMON:  # 单行 未定义白色
            img = np.array(Image.new("RGB", (440, 140), (255, 255, 255)))
            offset = 15
            img[25:115, offset:offset+45] = self._draw_char(plate[0])
            offset = offset + 45 + 34
            for i in range(1, len(plate)-1):
                img[25:115, offset:offset+45] = self._draw_char(plate[i])
                offset = offset + 45 + 12
        elif bg_type==White_Type.JINGCHA:  # 单行警牌
            img = np.array(Image.new("RGB", (440, 140), (255, 255, 255)))
            if (plate[-1]!='警'):
                raise NotImplementedError
            offset = 15
            img[25:115, offset:offset+45] = self._draw_char(plate[0])
            offset = offset + 45 + 34
            for i in range(1, len(plate)-1):
                img[25:115, offset:offset+45] = self._draw_char(plate[i])
                offset = offset + 45 + 12
        elif bg_type==White_Type.JUNDUI:  # 单行军牌
            img = np.array(Image.new("RGB", (490, 140), (255, 255, 255)))
            offset = 17
            img[28:112, offset:offset+50] = self._draw_char_jd(plate[0])
            offset = offset + 50 + 12
            img[28:112, offset:offset+50] = self._draw_char_jd(plate[1])
            offset = offset + 50 + 45
            for i in range(2, len(plate)):
                img[28:112, offset:offset+50] = self._draw_char_jd(plate[i])
                offset = offset + 50 + 12

            if p_front:     #前车牌点号红色
                imgcrop=img[28:112,15:169,:]
                imgcrop1 = imgcrop.sum(axis=2)
                imgcrop[imgcrop1<200]=[0,0,255]
                img[28:112,15:169]=imgcrop
            else:
                imgcrop=img[28:112,15:67,:]
                imgcrop1 = imgcrop.sum(axis=2)
                imgcrop[imgcrop1<200]=[0,0,255]
                img[28:112,15:67]=imgcrop
        elif bg_type==White_Type.WUJIN:  # 单行武警牌
            img = np.array(Image.new("RGB", (480, 140), (255, 255, 255)))
            offset = 15
            img[25:115, offset:offset+43] = self._draw_char_yj(plate[0])
            offset = offset + 43 + 8
            img[25:115, offset:offset+43] = self._draw_char_yj(plate[1])
            offset = offset + 43 + 35
            for i in range(2, len(plate)):
                img[25:115, offset:offset+43] = self._draw_char_yj(plate[i])
                offset = offset + 43 + 12
            if p_front:     #前车牌点号红色
                imgcrop=img[25:115,15:110,:]
                imgcrop1 = imgcrop.sum(axis=2)
                imgcrop[imgcrop1<200]=[0,0,255]
                img[25:115,15:110]=imgcrop
            else:
                imgcrop=img[25:115,15:197,:]
                imgcrop1 = imgcrop.sum(axis=2)
                imgcrop[imgcrop1<200]=[0,0,255]
                img[25:115,15:197]=imgcrop
        elif bg_type==White_Type.YINGJI:  # 应急车牌
            img = np.array(Image.new("RGB", (480, 140), (255, 255, 255)))
            if ((plate[-1]!='急')|(plate[-2]!='应')):
                raise NotImplementedError
            offset = 15
            img[25:115, offset:offset+45] = self._draw_char(plate[0])
            offset = offset + 45 + 31
            for i in range(1, len(plate)-2):
                img[25:115, offset:offset+43] = self._draw_char_yj(plate[i])
                offset = offset + 43 + 12
            if p_front:     #前车牌点号红色
                imgcrop=img[25:115,86:139,:]
                imgcrop1 = imgcrop.sum(axis=2)
                imgcrop[imgcrop1<200]=[0,0,255]
                img[25:115,86:139]=imgcrop
        elif bg_type==White_Type.JUNDUI_M:  # 双行军牌
            img = np.array(Image.new("RGB", (495, 240), (255, 255, 255)))
            offset = 118
            img[30:100, offset:offset+100] = self._draw_char_multi(plate[0])
            offset = offset + 100 + 60
            img[30:100, offset:offset+100] = self._draw_char_multi(plate[1])
            offset = 25
            for i in range(2, len(plate)):
                img[120:220, offset:offset+73] = self._draw_char_multi(plate[i],upper=False)
                offset = offset + 73 + 20
            if p_front:     #前车牌点号红色
                imgcrop=img[30:100,110:220,:]
                imgcrop1 = imgcrop.sum(axis=2)
                imgcrop[imgcrop1<200]=[0,0,255]
                img[30:100,110:220]=imgcrop
            else:
                imgcrop=img[30:100,110:380,:]
                imgcrop1 = imgcrop.sum(axis=2)
                imgcrop[imgcrop1<200]=[0,0,255]
                img[30:100,110:380]=imgcrop
        elif bg_type==White_Type.WUJIN_M:  # 双行武警牌
            img = np.array(Image.new("RGB", (495, 240), (255, 255, 255)))
            offset = 118
            img[30:100, offset:offset+73] = self._draw_char_wj_multi(plate[0])
            offset = offset + 73+4
            img[30:100, offset:offset+56] = self._draw_char_wj(plate[1])
            offset = offset + 56+20+15+20
            img[30:100, offset:offset+73] = self._draw_char_wj_multi(plate[2])
            offset = 25
            for i in range(3, len(plate)):
                img[120:220, offset:offset+73] = self._draw_char_wj_multi(plate[i],upper=False)
                offset = offset + 73 + 20

            if p_front:     #前车牌点号红色
                imgcrop=img[30:100,110:295,:]
                imgcrop1 = imgcrop.sum(axis=2)
                imgcrop[imgcrop1<200]=[0,0,255]
                img[30:100,110:295]=imgcrop
            else:
                imgcrop=img[30:100,110:440,:]
                imgcrop1 = imgcrop.sum(axis=2)
                imgcrop[imgcrop1<200]=[0,0,255]
                img[30:100,110:440]=imgcrop
        else:
            raise NotImplementedError
        return img

    def _draw_bg(self, bg_type):
        if bg_type == White_Type.COMMON:
            bg = random.choice(self._bg_common)
        elif bg_type==White_Type.JINGCHA:
            bg = random.choice(self._bg_jingcha)  
        elif bg_type==White_Type.JUNDUI:
            bg = random.choice(self._bg_jundui)   
        elif bg_type==White_Type.JUNDUI_M:
            bg = random.choice(self._bg_jundui_m)   
        elif bg_type==White_Type.WUJIN:
            bg = random.choice(self._bgs_wujing)
        elif bg_type==White_Type.WUJIN_M:
            bg = random.choice(self._bgs_wujing_m)
        elif bg_type==White_Type.YINGJI:
            bg = random.choice(self._bgs_yingji)
        else:
            raise NotImplementedError
        return bg


if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="Generate a black plate.")
    parser.add_argument("plate", help="license plate number (default: 京A12345)", type=str, nargs="?", default="WJX30522")
    args = parser.parse_args()

    draw = Draw()
    plate = draw(args.plate, bg_type=5)  #bg_type=0，单行'警'牌，军牌、双行待添加
    plt.imshow(plate)
    plt.show()
