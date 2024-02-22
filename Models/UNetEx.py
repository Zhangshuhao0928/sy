import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from Models.AutoEncoder import create_layer
from typing import List, Tuple, Dict


# 创建一层 encoder，用于压缩特征
def create_encoder_block(in_channels: int, out_channels: int, kernel_size: int, wn: bool = True, bn: bool = True,
                         activation: nn = nn.ReLU, layers: int = 2) -> nn.Sequential:
    """
    @param in_channels: 输入通道维度尺寸
    @param out_channels: 输出通道维度尺寸
    @param kernel_size: 卷积核尺寸
    @param wn: 权重归一化 bool 变量
    @param bn: 批量归一化 bool 变量
    @param activation: 激活函数选择
    @param layers: 一个 encoder 包含多少个 multi-layers
    @return: 一层 encoder 网络结构，nn.Sequential 对象
    """

    encoder_block = []
    # 如果当前层是第一层，那么输入维度为 in_channels，输出维度为 out_channels
    # 如果不是第一层，那么输入维度为了承接前面的层的输出尺寸，为 out_channels，输出维度一直都是 out_channels
    for i in range(layers):
        _in = out_channels
        _out = out_channels
        if i == 0:
            _in = in_channels

        encoder_block.append(create_layer(_in, _out, kernel_size, wn, bn, activation, nn.Conv2d))

    # 将 list 对象（list 中的每一个元素都是一个nn.Sequential对象）转换成 nn.Sequential对象
    # 转换之后的返回值为一个大的 nn.Sequential对象，其中包含多个小的 nn.Sequential对象
    return nn.Sequential(*encoder_block)


# 创建一层 decoder，用于回溯放大特征
def create_decoder_block(in_channels: int, out_channels: int, kernel_size: int, wn: bool = True, bn: bool = True,
                         activation: nn = nn.ReLU, layers: int = 2, final_block: bool = False) -> nn.Sequential:
    """
    @param in_channels: 输入通道维度尺寸
    @param out_channels: 输出通道维度尺寸
    @param kernel_size: 卷积核尺寸
    @param wn: 权重归一化 bool 变量
    @param bn: 批量归一化 bool 变量
    @param activation: 激活函数选择
    @param layers: 一个 decoder 包含多少个 multi-layers
    @param final_block: 判断是否是最后一个 decoder_block
    @return: 一个 decoder_block，nn.Sequential 对象
    """

    decoder_block = []
    # 这里每个 decoder_block 中的 layer 是按照顺序创建的，不是逆序
    for i in range(layers):
        # 中间层的输入和输出都是 in_channels
        _in = in_channels
        _out = in_channels
        _bn = bn
        _activation = activation
        # 如果是 decoder_block 中的第一层，那么其 input 通道维度应该为前一个 decoder_block 的 output 通道维度
        # 因为 filters 中的设计是 8 16 32 64 所以 decoder 部分是 64 32 16 8
        # 即下一层 decoder_block 的 input 通道*2 = 上一层 decoder_block 的 output 通道
        if i == 0:
            _in = in_channels * 2
        # 如果是 decoder_block 中的最后一层，将 output 设置为 out_channels
        # 如果是最后一个 decoder_block，则将最后的批量归一化以及激活函数置为None（空）
        if i == layers - 1:
            _out = out_channels
            if final_block:
                _bn = False
                _activation = None

        # 由于 decoder_block 中的 layer 是按顺序创建的，所以这里直接按照顺序 append 即可，无需逆序
        decoder_block.append(create_layer(_in, _out, kernel_size, wn, _bn, _activation, nn.ConvTranspose2d))

    return nn.Sequential(*decoder_block)


# 创建整个 encoder 网络结构
def create_encoder(in_channels: int, filters: List, kernel_size: int, wn: bool = True, bn: bool = True,
                   activation: nn = nn.ReLU, layers: int = 2) -> nn.Sequential:
    """
    @param in_channels: 输入通道维度尺寸
    @param filters: 卷积核通道维度列表
    @param kernel_size: 卷积核尺寸
    @param wn: 权重归一化 bool 变量
    @param bn: 批量归一化 bool 变量
    @param activation: 激活函数选择
    @param layers: 一个 encoder 包含多少个 multi-layers
    @return: 整个 encoder 网络结构，nn.Sequential 对象，里面的每个对象是一层 encoder，也是 nn.Sequential 对象，其中的每一层
             仍然是 nn.Sequential对象，所以是三层的 nn.Sequential 对象
    """

    encoder = []
    # 这里的主要逻辑是网络尺寸的衔接
    # 当创建第 1 层 encoder 时，输入通道维度为 in_channels，输出通道维度为 filters[0]
    # 那么创建第 2 层 encoder 时，输出通道维度为前面输出的 filters[0], 输出通道维度为 filters[1]
    # 以此类推
    for i in range(len(filters)):
        if i == 0:
            encoder_layer = create_encoder_block(in_channels, filters[i], kernel_size, wn, bn, activation, layers)
        else:
            encoder_layer = create_encoder_block(filters[i - 1], filters[i], kernel_size, wn, bn, activation, layers)

        encoder.append(encoder_layer)

    return nn.Sequential(*encoder)


# 创建整个 decoder 网络结构
def create_decoder(out_channels: int, filters: List, kernel_size: int, wn: bool = True, bn: bool = True,
                   activation: nn = nn.ReLU, layers: int = 2) -> nn.Sequential:
    """
    @param out_channels: 输出通道维度尺寸，目前这里被默认为指定为 1
    @param filters: 卷积核通道维度列表
    @param kernel_size: 卷积核尺寸
    @param wn: 权重归一化
    @param bn: 批量归一化
    @param activation: 激活函数选择
    @param layers: 一个 decoder 包含多少个 multi-layers
    @return: 整个 decoder 网络结构，nn.Sequential 对象
    """

    # 这里的逻辑和创建 encoder 的逻辑基本一致
    # 区别就是 decoder 是反向创建的，encoder 是先创建第 1 个 encoder_layer，然后创建第二个 encoder_layer
    # decoder 这里先创建最后一个 decoder_layer，然后再依次创建前面的 decoder_layer
    # 因为数据的流动是 经过 encoder 8->16->32->64,所以第一个 decoder_block 应该是 64，即 64->32->16->8
    decoder = []
    for i in range(len(filters)):
        # i=0 表明是最后一层 decoder_layer, 因为 filters[0]是 8
        if i == 0:
            decoder_layer = create_decoder_block(filters[i], out_channels, kernel_size, wn, bn, activation, layers,
                                                 final_block=True)
        else:
            decoder_layer = create_decoder_block(filters[i], filters[i - 1], kernel_size, wn, bn, activation, layers,
                                                 final_block=False)
        # 这里正常从左到右顺序 append 的，即最后一层在第一个位置，后续需要反转 list
        decoder.append(decoder_layer)

    # 反转 decoder 使其顺序为逆袭
    decoder = decoder[::-1]

    return nn.Sequential(*decoder)


# UNet 类
class UNetEx(nn.Module):
    # 网络初始化
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, filters: List = [16, 32, 64],
                 layers: int = 3, weight_norm: bool = True, batch_norm: bool = True, activation: nn = nn.ReLU,
                 final_activation: nn = None) -> None:
        """
        @param in_channels: 输入通道维度尺寸
        @param out_channels: 输出通道维度尺寸
        @param kernel_size: 卷积核尺寸
        @param filters: 卷积核通道维度
        @param layers: 每个 encoder_block 以及 decoder_block 包含的网络层数
        @param weight_norm: 权重归一化
        @param batch_norm: 批量归一化
        @param activation: 激活函数选择
        @param final_activation: 最后的激活函数，用于输出结果前一刻
        """

        # 继承基类 nn.Module 的初始化方法
        super().__init__()
        # 断言，判断卷积核的数目是否＞0，≤0会报错
        assert len(filters) > 0
        self.final_activation = final_activation
        # 创建 encoder
        self.encoder = create_encoder(in_channels, filters, kernel_size, weight_norm, batch_norm, activation, layers)

        # 创建 decoder，这里根据 out_channels 分别创建了 out_channels 个 decoder，每个 decoder 最终的输出通道维度是 1
        # 其实可以直接指定 decoder 的输出通道维度为 out_channels，这样只创建一个 decoder 即可，从理论上来说应该是没啥区别的
        # 后续跑实验的时候测试一下
        decoders = []
        for i in range(out_channels):
            decoders.append(create_decoder(1, filters, kernel_size, weight_norm, batch_norm, activation, layers))
        self.decoders = nn.Sequential(*decoders)

    # encode process
    def encode(self, x: torch.Tensor) -> Tuple:
        """
        @param x: batch data tensors
        @return:
            x: encoder 编码出的高维特征向量
            intermedia_tensors: 记录 encoder 中每个 encoder_block 的中间输出结果
            indices: 记录max_pooling 时选取的 2x2 网格中的下标，用于 decoder 上采样时复原尺寸
            sizes: 记录 encoder 中每个 encoder_block 的中间输出结果的尺寸
        """

        # 用于存储中间过程每一个 encoder_block 输出的中间结果
        intermedia_tensors = []
        # 用于记录最大池化层下采样时选取的下标索引，因为 encoder 和 decoder 结构是对称的
        # 后面会用这个索引在 decoder 中进行上采样还原尺寸
        indices = []
        # 记录中间结果的尺寸
        sizes = []

        for encoder_block in self.encoder:
            x = encoder_block(x)
            sizes.append(x.size())
            intermedia_tensors.append(x)
            x, ind = F.max_pool2d(x, 2, 2, return_indices=True)
            indices.append(ind)

        return x, intermedia_tensors, indices, sizes

    # decode process
    def decode(self, _x: torch.Tensor, _tensors: List, _indices: List, _sizes: List) -> torch.Tensor:
        """
        @param _x: encoder 输出的高维特征向量
        @param _tensors: encoder 编码过程中的中间结果变量列表
        @param _indices: encoder 下采样 max_pooling 时的下标列表
        @param _sizes: encoder 编码过程中的中间结果的尺寸列表
        @return: output, size: (batch_size, out_channels, 50, 250)
        """

        y = []
        # 每个 decoder 分别解码得到各自的输出
        for _decoder in self.decoders:
            # 由于有多个 decoder，所以用于每个 decoder 解码的所有数据都是一样的
            x = _x
            tensors = _tensors[:]
            indices = _indices[:]
            sizes = _sizes[:]

            for decoder_block in _decoder:
                # pop方法从 list 列表从后向前弹出数据，以此跟 encoder 对称
                tensor = tensors.pop()
                size = sizes.pop()
                ind = indices.pop()
                x = F.max_unpool2d(x, ind, 2, 2, output_size=size)
                # 拼接 encoder 的中间变量以及 decoder 的中间输出
                x = torch.cat([tensor, x], dim=1)
                x = decoder_block(x)

            y.append(x)

        # 在通道维度拼接
        return torch.cat(y, dim=1)

    # 网络的前向计算过程 forward process
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        @param x: batch data tensors
        @return: output
        """

        # encode
        x, intermedia_tensors, indices, sizes = self.encode(x)
        # decode
        x = self.decode(x, intermedia_tensors, indices, sizes)
        # 根据 final_activation 添加最后一个激活函数
        if self.final_activation is not None:
            x = self.final_activation(x)

        return x
