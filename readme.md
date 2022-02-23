### 车牌识别

车牌识别部分，参考[LPRNet](https://github.com/sirius-ai/LPRNet_Pytorch)和[Multi-line license plate recognition](https://github.com/deeplearningshare/multi-line-plate-recognition)，

支持车牌种类
 - 蓝色单层车牌
 - 黄色单层车牌
 - 绿色新能源车牌、民航车牌
 - 黑色单层车牌
 - 白色警牌、军牌、武警车牌
 - 黄色双层车牌
 - 绿色农用车牌
 - 白色双层军牌

理论上该网络可以识别单双行，只要添加数据训练，当前训练数据均为生成数据

数据来源
 - CCPD
 - 生成数据参考[fake_chs_lp](https://github.com/ufownl/fake_chs_lp)、[alpr_utils](https://github.com/ufownl/alpr_utils)

车牌检测参考：https://github.com/gm19900510/Pytorch_Retina_License_Plate