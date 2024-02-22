import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


# 创建每个 encoder block 中一层 encoder 中的的 multi-layers
def create_layer(in_channels: int, out_channels: int, kernel_size: int, wn: bool = True, bn: bool = True,
                 activation: nn = nn.ReLU, convolution: nn = nn.Conv2d) -> nn.Sequential:
    """
        @param in_channels: 输入通道维度尺寸
        @param out_channels: 输出通道维度尺寸
        @param kernel_size: 卷积核尺寸
        @param wn: 权重归一化 bool 变量
        @param bn: 批量归一化 bool 变量
        @param activation: 激活函数选择
        @param convolution: 卷积层选择
        @return: block 中的一层网络结构，nn.Sequential 对象
        """

    # 断言判断卷积核尺寸是否是奇数，偶数报错
    assert kernel_size % 2 == 1
    layer = []
    # 这里的 convolution 就是 nn.Conv2d 或者 nn.ConvTranspose2d
    conv = convolution(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
    # 如果启用权重归一化，则对于 conv 层的参数进行归一化
    if wn:
        conv = weight_norm(conv)
    layer.append(conv)
    # 这里的 activation 就是 nn.ReLU
    if activation is not None:
        # 在每一个卷积层后面添加一个激活层
        layer.append(activation())
    # 如果启用批量归一化，则在输出维度上做 BN
    if bn:
        layer.append(nn.BatchNorm2d(out_channels))

    # 将 list 对象转换成 torch 的 sequential 对象，形成一个串行的网络 block
    return nn.Sequential(*layer)


class AutoEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, filters=[16, 32, 64],
                 weight_norm=True, batch_norm=True, activation=nn.ReLU, final_activation=None):
        super().__init__()
        assert len(filters) > 0
        encoder = []
        decoder = []
        for i in range(len(filters)):
            if i == 0:
                encoder_layer = create_layer(in_channels, filters[i], kernel_size, weight_norm, batch_norm, activation, nn.Conv2d)
                decoder_layer = create_layer(filters[i], out_channels, kernel_size, weight_norm, False, final_activation, nn.ConvTranspose2d)
            else:
                encoder_layer = create_layer(filters[i-1], filters[i], kernel_size, weight_norm, batch_norm, activation, nn.Conv2d)
                decoder_layer = create_layer(filters[i], filters[i-1], kernel_size, weight_norm, batch_norm, activation, nn.ConvTranspose2d)
            encoder = encoder + [encoder_layer]
            decoder = [decoder_layer] + decoder
        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):
        return self.decoder(self.encoder(x))
