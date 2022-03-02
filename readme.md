<!--
 * @Date: 2022-01-05 22:53:35
 * @Author: ChHanXiao
 * @Github: https://github.com/ChHanXiao
 * @LastEditors: ChHanXiao
 * @LastEditTime: 2022-03-02 21:51:43
 * @FilePath: /license-plate-recoginition/readme.md
-->
### 车牌识别

车牌识别部分，参考[LPRNet](https://github.com/sirius-ai/LPRNet_Pytorch)和[Multi-line license plate recognition](https://github.com/deeplearningshare/multi-line-plate-recognition)

车牌数据生成，参考[fake_chs_lp](https://github.com/ufownl/fake_chs_lp)、[alpr_utils](https://github.com/ufownl/alpr_utils)

理论上该网络可以识别单双行，只要添加数据训练，当前训练数据均为生成数据

数据来源
 - CCPD2019，CCPD2020，裁切矫正车牌，参考[prepare_ccpd.py](datasets/prepare_ccpd.py)
 - 生成数据，通过[gen_data.py](gen_data.py)生成数据集，每类车牌及细分类别的生成概率可通过配置文件[config_gen.py](data/config_gen.py)配置
 - 生成数据有使用贴图，可在`data/env_imgs`中添加背景素材
 - 车牌标签以文件名方式保存和读取，'日期'-'种类'-'号码'-'时间戳'，如 `20220122-green+b-藏AQ2580D-1642860498000912.jpg`

训练
 - 修改训练的参数配置文件[lpr.yml](config/lpr.yml)，一般修改保存路径、train和test数据路径，因为标签以文件名方式读取，只需要添加文件夹路径即可
 - 修改配置文件后，`python train.py config/lpr.yml`开始训练
 - 建议单卡训练（多卡eval未修改），小的BatchSize对指标并无影响
测试
 - 评估指标 `python test.py --task val --config lpr-d.yml --model path/to/your/model`
 - 结果测试 `python inference.py --task val --config lpr-d.yml --model path/to/your/model --path path/to/your/img`

支持车牌种类
 - 蓝色单层车牌
 - 黄色单层车牌
 - 绿色新能源车牌、民航车牌
 - 黑色单层车牌
 - 白色警牌、军牌、武警车牌
 - 黄色双层车牌
 - 绿色农用车牌
 - 白色双层军牌

车牌检测参考：https://github.com/gm19900510/Pytorch_Retina_License_Plate