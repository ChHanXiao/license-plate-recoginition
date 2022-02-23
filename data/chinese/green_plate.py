import os
import cv2
import numpy as np
from PIL import Image, ImageDraw
import random
from enum import Enum
class Green_Type(Enum):
    SMALL = 0
    LARGE = 1

def load_font():
    return {
        "京": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/ne000.png")),
        "津": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/ne001.png")),
        "冀": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/ne002.png")),
        "晋": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/ne003.png")),
        "蒙": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/ne004.png")),
        "辽": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/ne005.png")),
        "吉": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/ne006.png")),
        "黑": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/ne007.png")),
        "沪": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/ne008.png")),
        "苏": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/ne009.png")),
        "浙": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/ne010.png")),
        "皖": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/ne011.png")),
        "闽": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/ne012.png")),
        "赣": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/ne013.png")),
        "鲁": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/ne014.png")),
        "豫": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/ne015.png")),
        "鄂": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/ne016.png")),
        "湘": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/ne017.png")),
        "粤": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/ne018.png")),
        "桂": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/ne019.png")),
        "琼": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/ne020.png")),
        "渝": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/ne021.png")),
        "川": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/ne022.png")),
        "贵": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/ne023.png")),
        "云": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/ne024.png")),
        "藏": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/ne025.png")),
        "陕": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/ne026.png")),
        "甘": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/ne027.png")),
        "青": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/ne028.png")),
        "宁": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/ne029.png")),
        "新": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/ne030.png")),
        "A": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/ne100.png")),
        "B": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/ne101.png")),
        "C": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/ne102.png")),
        "D": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/ne103.png")),
        "E": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/ne104.png")),
        "F": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/ne105.png")),
        "G": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/ne106.png")),
        "H": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/ne107.png")),
        "J": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/ne108.png")),
        "K": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/ne109.png")),
        "L": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/ne110.png")),
        "M": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/ne111.png")),
        "N": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/ne112.png")),
        "P": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/ne113.png")),
        "Q": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/ne114.png")),
        "R": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/ne115.png")),
        "S": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/ne116.png")),
        "T": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/ne117.png")),
        "U": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/ne118.png")),
        "V": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/ne119.png")),
        "W": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/ne120.png")),
        "X": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/ne121.png")),
        "Y": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/ne122.png")),
        "Z": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/ne123.png")),
        "0": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/ne124.png")),
        "1": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/ne125.png")),
        "2": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/ne126.png")),
        "3": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/ne127.png")),
        "4": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/ne128.png")),
        "5": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/ne129.png")),
        "6": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/ne130.png")),
        "7": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/ne131.png")),
        "8": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/ne132.png")),
        "9": cv2.imread(os.path.join(os.path.dirname(__file__), "res/ne/ne133.png"))
    }

class Draw:
    def __init__(self):
        self.lplen = [8,]
        self._font = load_font()
        self._bg_s = [
            cv2.resize(cv2.imread(os.path.join(os.path.dirname(__file__), f"res/green/green_bg_s_{imgnum}.png")), (480, 140)) for imgnum in np.arange(3)
            ]
        self._bg_l = [
            cv2.resize(cv2.imread(os.path.join(os.path.dirname(__file__), f"res/green/green_bg_g_{imgnum}.png")), (480, 140)) for imgnum in np.arange(4)
            ]

    def __call__(self, plate, bg_type=Green_Type.SMALL):
        if len(plate) not in self.lplen:
            print("ERROR: Invalid length")
            return None

        bg = self._draw_bg(bg_type)
        fg = self._draw_fg(plate, bg_type)
        return cv2.cvtColor(cv2.bitwise_and(fg, bg), cv2.COLOR_BGR2RGB)
        
    def _draw_char(self, ch):
        return cv2.resize(self._font[ch], (43 if ch.isupper() or ch.isdigit() else 45, 90))

    def _draw_fg(self, plate, bg_type):
        img = np.array(Image.new("RGB", (480, 140), (255, 255, 255)))
        offset = 15
        img[25:115, offset:offset+45] = self._draw_char(plate[0])
        offset = offset + 45 + 9
        img[25:115, offset:offset+43] = self._draw_char(plate[1])
        offset = offset + 43 + 49
        for i in range(2, len(plate)):
            img[25:115, offset:offset+43] = self._draw_char(plate[i])
            offset = offset + 43 + 9
        return img

    def _draw_bg(self, bg_type):
        if bg_type==Green_Type.SMALL:
            bg = random.choice(self._bg_s)
        elif bg_type==Green_Type.LARGE:
            bg = random.choice(self._bg_l)
        else:
            raise NotImplementedError
        return bg

if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="Generate a green plate.")
    parser.add_argument("plate", help="license plate number (default: 京AD12345)", type=str, nargs="?", default="京AD12345")
    args = parser.parse_args()

    draw = Draw()
    plate = draw(args.plate, bg_type=1)  #bg_type=0，渐变绿，bg_type=1，黄绿双色
    plt.imshow(plate)
    plt.show()
