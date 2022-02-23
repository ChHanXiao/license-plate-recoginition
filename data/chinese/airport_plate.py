import os
import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import random

from enum import Enum
class Airport_Type(Enum):
    COMMON = 0

def load_font():
    return {
        # "京": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_京.jpg")),
        # "津": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_津.jpg")),
        # "冀": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_冀.jpg")),
        # "晋": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_晋.jpg")),
        # "蒙": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_蒙.jpg")),
        # "辽": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_辽.jpg")),
        # "吉": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_吉.jpg")),
        # "黑": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_黑.jpg")),
        # "沪": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_沪.jpg")),
        # "苏": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_苏.jpg")),
        # "浙": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_浙.jpg")),
        # "皖": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_皖.jpg")),
        # "闽": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_闽.jpg")),
        # "赣": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_赣.jpg")),
        # "鲁": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_鲁.jpg")),
        # "豫": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_豫.jpg")),
        # "鄂": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_鄂.jpg")),
        # "湘": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_湘.jpg")),
        # "粤": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_粤.jpg")),
        # "桂": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_桂.jpg")),
        # "琼": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_琼.jpg")),
        # "渝": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_渝.jpg")),
        # "川": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_川.jpg")),
        # "贵": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_贵.jpg")),
        # "云": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_云.jpg")),
        # "藏": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_藏.jpg")),
        # "陕": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_陕.jpg")),
        # "甘": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_甘.jpg")),
        # "青": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_青.jpg")),
        # "宁": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_宁.jpg")),
        # "新": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_新.jpg")),
		# "澳": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_澳.jpg")),
		# "港": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_港.jpg")),
        # "学": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_学.jpg")),
        # "警": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/140_警.jpg")),
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

class Draw:
    def __init__(self):
        self.lplen = [7,]
        self._font = load_font()
        self._bg = [
            cv2.resize(cv2.imread(os.path.join(os.path.dirname(__file__), f"res/airport/c_{imgnum}.png")), (440, 140)) for imgnum in np.arange(1)
            ]
    def __call__(self, plate, bg_type=Airport_Type.COMMON):
        if len(plate) not in self.lplen:
            print("ERROR: Invalid length")
            return None
        bg = self._draw_bg(bg_type)
        fg = self._draw_fg(plate, bg_type)
        return cv2.cvtColor(cv2.bitwise_or(fg, bg), cv2.COLOR_BGR2RGB)

    def _draw_char(self, ch,):
        return 255-cv2.resize(self._font[ch], (45, 90))

    def _draw_fg(self, plate, bg_type):
        img = np.array(Image.new("RGB", (440, 140), (0, 0, 0)))
        if bg_type==Airport_Type.COMMON:
            if (plate[0]!='民') | (plate[1]!='航'):
                raise NotImplementedError
            offset = 15
            # img[25:115, offset:offset+45] = self._draw_char(plate[0])
            offset = offset + 45 + 12
            # img[25:115, offset:offset+45] = self._draw_char(plate[1])
            offset = offset + 45 + 34
            for i in range(2, len(plate)):
                img[25:115, offset:offset+45] = self._draw_char(plate[i])
                offset = offset + 45 + 12
        else:
            raise NotImplementedError
        return img

    def _draw_bg(self, bg_type):
        if bg_type==Airport_Type.COMMON:
            bg = random.choice(self._bg)
        else:
            raise NotImplementedError
        return bg

if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    parser = argparse.ArgumentParser(description="Generate a black plate.")
    parser.add_argument("plate", help="license plate number (default: 民航D4345)", type=str, nargs="?", default="粤A1234澳")
    args = parser.parse_args()
    draw = Draw()
    plate = draw(args.plate, bg_type=Airport_Type.COMMON) 
    plt.imshow(plate)
    plt.show()
