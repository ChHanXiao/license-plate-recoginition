from pickle import TRUE
from sys import path
import torch.nn as nn
import torch
import torch.nn.functional as F

def sparse_tuple_for_ctc(T_length, lengths):
    input_lengths = []
    target_lengths = []

    for ch in lengths:
        input_lengths.append(T_length)
        target_lengths.append(ch)

    return tuple(input_lengths), tuple(target_lengths)

class small_basic_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(small_basic_block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out, kernel_size=1),
        )
    def forward(self, x):
        return self.block(x)

# ============== 双层车牌识别 ==============
# 基于lprnet和Multi-line license plate recognition双层车牌识别网络
class PlateNet_multi(nn.Module):
    def __init__(self, class_num, dropout_rate=0.5):
        super(PlateNet_multi, self).__init__()
        self.class_num = class_num
        self.ctc_loss = nn.CTCLoss(blank=class_num-1, reduction='mean')
        self.sparse_tuple = sparse_tuple_for_ctc

        self.dropout_rate = dropout_rate
        self.stage1 = nn.Sequential(                                                                            # 3x48x96
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),                  # 64x48x96
                nn.MaxPool2d((2, 2), stride=(2, 2), ceil_mode=True),                                            # 64x24x48
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2, groups=64),      # 64x24x48
                nn.BatchNorm2d(num_features=64),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=(2,0), groups=64),  # 64x24x44  S
                nn.BatchNorm2d(num_features=64),
                nn.ReLU(),
                small_basic_block(ch_in=64, ch_out=128),                                                        # 128x24x44
                nn.BatchNorm2d(num_features=128),
                nn.ReLU()
                )
        self.outconv1 = nn.Sequential(                                                                          # 128x24x44 k(5,5) s(5,2)   Stage1
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=(5,2), groups=128),          # 128x4x20  k(3,3) s(1,1)
                nn.BatchNorm2d(num_features=128),
                nn.ReLU()
                )
        self.down1 = nn.Sequential(
                small_basic_block(ch_in=128, ch_out=128),                                                    # 128x24x44 
                nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1)                                   # 64x24x44
                )                                              
        self.stage2 = nn.Sequential(                                                                            # 64x24x44  k(5,5) s(2,2) p(2,2) 
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=2, padding=2, groups=64),      # 64x12x22  S
                nn.BatchNorm2d(num_features=64),
                nn.ReLU(),
                small_basic_block(ch_in=64, ch_out=128),                                                        # 128x12x22 S
                nn.BatchNorm2d(num_features=128),
                nn.ReLU()
                )
        self.outconv2 = nn.Sequential(                                                                        # 128x12x22 k(5,3) s(2,1)   Stage2
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5,3), stride=(2,1), groups=128),      # 128x4x20  k(3,3) s(1,1)
                nn.BatchNorm2d(num_features=128),
                nn.ReLU()
                )
        self.down2 = nn.Sequential(
                small_basic_block(ch_in=128, ch_out=128),                                                     # 128x12x22 
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1)                                   # 128x12x22
                )   
        self.stage3 = nn.Sequential(                                                                                 # 128x12x22  
                small_basic_block(ch_in=128, ch_out=128),                                                                # 128x12x22 k(1,1) s(1,1)
                nn.BatchNorm2d(num_features=128),
                nn.ReLU(),
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=1),                                  # 256x12x22  
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,13), stride=(2,1), padding=(1,6), groups=256),  # 256x6x22  k(3,3) s(1,1)
                nn.BatchNorm2d(num_features=256),
                nn.ReLU(),
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, groups=256),                      # 256x4x20  k(3,3) s(1,1)
                nn.BatchNorm2d(num_features=256),
                nn.ReLU(),
                )  
        self.container = nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, groups=512),          # 512x2x18
                nn.BatchNorm2d(num_features=512),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                )
        self.container2 = nn.Conv2d(in_channels=512, out_channels=class_num, kernel_size=(1, 17), stride=(1, 2),padding=(0,8))
        self._init_weight()

    def forward(self, x):               # 3x48x96  
        out1 = self.stage1(x)           # 128x24x44
        out = self.down1(out1)          # 64x24x44
        out2 = self.stage2(out)         # 128x12x22
        out = self.down2(out2)          # 128x12x22 
        out3 = self.stage3(out)         # 256x4x20     
        out1 = self.outconv1(out1)      # 128x4x20
        out2 = self.outconv2(out2)      # 128x4x20
        logits = torch.cat((out1, out2, out3), 1)   # 512x4x20
        logits = self.container(logits)             # 512x2x18
        top, bottom=torch.split(logits,1,dim=2)
        logits = torch.cat((top, bottom), 3)        # 512x1x36
        logits = self.container2(logits)            # cx1x18
        # logits = torch.Tensor.squeeze(logits, dim=2) # nxcx18
        logits = logits.permute(0, 3, 1, 2)         # 18xcx1
        logits = torch.Tensor.squeeze(logits, dim=3)# nx18xc
        return logits

    def _init_weight(self):
        def xavier(param):
            nn.init.xavier_uniform(param)
        def weights_init(m):
            for key in m.state_dict():
                if key.split('.')[-1] == 'weight':
                    if 'conv' in key:
                        nn.init.kaiming_normal_(m.state_dict()[key], mode='fan_out')
                    if 'bn' in key:
                        m.state_dict()[key][...] = xavier(1)
                elif key.split('.')[-1] == 'bias':
                    m.state_dict()[key][...] = 0.01
        self.stage1.apply(weights_init)
        self.stage2.apply(weights_init)
        self.stage3.apply(weights_init)
        self.down1.apply(weights_init)
        self.down2.apply(weights_init)
        self.outconv1.apply(weights_init)
        self.outconv2.apply(weights_init)
        self.container.apply(weights_init)
        self.container2.apply(weights_init)
        print("initial net weights successful!")

# 基于PlateNet_multi 裁剪
class PlateNet_multi_1(nn.Module):
    def __init__(self, class_num, dropout_rate=0.5):
        super(PlateNet_multi_1, self).__init__()
        self.class_num = class_num
        self.ctc_loss = nn.CTCLoss(blank=class_num-1, reduction='mean')
        self.sparse_tuple = sparse_tuple_for_ctc
        self.dropout_rate = dropout_rate
        self.stage1 = nn.Sequential(                                                         # 3x48x96
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),   # 64x48x96
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=(2, 2), ceil_mode=True),                             # 64x24x48
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1,5), stride=1),         # 64x24x44 
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            small_basic_block(ch_in=64, ch_out=64),                                          # 64x24x44
            nn.BatchNorm2d(num_features=64),
            nn.ReLU()
            )
        self.outconv1 = nn.Sequential(                                                       # 64x24x44
            small_basic_block(ch_in=64, ch_out=128),                                         # 128x24x44 
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d((6, 6), stride=(6, 2), ceil_mode=True),                             # 128x4x20
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1),           # 128x4x20
            nn.ReLU()
            )
        self.stage2 = nn.Sequential(                                                         # 64x24x44 
            small_basic_block(ch_in=64, ch_out=128),                                         # 128x24x44 
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=(2, 2), ceil_mode=True),                             # 128x12x22 
            small_basic_block(ch_in=128, ch_out=128),                                        # 128x12x22
            nn.BatchNorm2d(num_features=128),
            nn.ReLU()
            )
        self.outconv2 = nn.Sequential(                                                       # 128x12x22
            small_basic_block(ch_in=128, ch_out=256),                                        # 256x12x22 
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool2d((6, 3), stride=(2, 1), ceil_mode=True),                             # 256x4x20
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1), stride=(1, 1)), # 256x4x20
            nn.ReLU()
            )   
        self.container1 = nn.Sequential(                                                       # (128+256)x4x20
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, groups=384), # 384x2x18
            nn.BatchNorm2d(num_features=384),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            )
        self.container2 = nn.Conv2d(in_channels=384, out_channels=class_num, kernel_size=(1, 17), stride=(1, 2),padding=(0,8))
        self._init_weight()

    def forward(self, x):                       # 3x48x96  
        out1 = self.stage1(x)                   # 64x24x44     
        out2 = self.stage2(out1)                # 128x12x22        
        out1 = self.outconv1(out1)              # 128x4x20
        out2 = self.outconv2(out2)              # 256x4x20
        logits = torch.cat((out1, out2), 1)     # 384x4x20
        logits = self.container1(logits)         # 384x2x18
        top, bottom=torch.split(logits, 1, dim=2)
        logits = torch.cat((top, bottom), 3)    # 384x1x36
        logits = self.container2(logits)        # cx1x18
        # logits = torch.Tensor.squeeze(logits, dim=2) # nxcx18
        logits = logits.permute(0, 3, 1, 2)         # 18xcx1
        logits = torch.Tensor.squeeze(logits, dim=3)# nx18xc
        return logits

    def _init_weight(self):
        def xavier(param):
            nn.init.xavier_uniform(param)
        def weights_init(m):
            for key in m.state_dict():
                if key.split('.')[-1] == 'weight':
                    if 'conv' in key:
                        nn.init.kaiming_normal_(m.state_dict()[key], mode='fan_out')
                    if 'bn' in key:
                        m.state_dict()[key][...] = xavier(1)
                elif key.split('.')[-1] == 'bias':
                    m.state_dict()[key][...] = 0.01
        self.stage1.apply(weights_init)
        self.stage2.apply(weights_init)
        self.outconv1.apply(weights_init)
        self.outconv2.apply(weights_init)
        self.container1.apply(weights_init)
        self.container2.apply(weights_init)
        print("initial net weights successful!")

if __name__ == "__main__":
    CHARS_S = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
            '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
            '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
            '新', '学', '港', '澳', '警', '使', '领', '应', '急',
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
            'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
            'W', 'X', 'Y', 'Z', '-'
            ]
    CHARS_DICT_S = {char:i for i, char in enumerate(CHARS_S)}
    CHARS_M =  ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
                '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
                '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
                '新', '学', '港', '澳', '警', '使', '领', '应', '急', '挂',
                '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
                'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
                'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
                'W', 'X', 'Y', 'Z', '-'
            ]
    CHARS_DICT_M = {char:i for i, char in enumerate(CHARS_M)}
    dummy_input = torch.randn(1,3,48,96)
    model = PlateNet_multi(len(CHARS_M))
    test_output = model(dummy_input)
    model.eval()
    onnx_path = 'lprnet.onnx'
    torch.onnx.export(model,dummy_input,onnx_path)
