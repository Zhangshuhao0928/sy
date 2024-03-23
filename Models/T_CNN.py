#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/3/18 15:53
# @Author : Erhaoo
# @Belief : Everything will be fine～

import torch

ACT_NAME = "tanh"  # tanh, relu


class Conv2d_Encoder(torch.nn.Module):
    def __init__(self, config=None,
                 act_name="relu"
                 ):
        super(Conv2d_Encoder, self).__init__()
        self.Conv2d_Layer = None
        if config is None:
            config = [{"in_channels": 4,
                       "out_channels": 8,
                       "kernel_size": (3, 3),
                       "stride": (1, 1),
                       "padding": (1, 1)},
                      {"in_channels": 8,
                       "out_channels": 16,
                       "kernel_size": (3, 3),
                       "stride": (1, 1),
                       "padding": (1, 1)},
                      {"in_channels": 16,
                       "out_channels": 32,
                       "kernel_size": (3, 3),
                       "stride": (1, 1),
                       "padding": (1, 1)},
                      {"in_channels": 32,
                       "out_channels": 64,
                       "kernel_size": (3, 3),
                       "stride": (1, 1),
                       "padding": (1, 1)},
                      {"in_channels": 64,
                       "out_channels": 128,
                       "kernel_size": (3, 3),
                       "stride": (1, 1),
                       "padding": (1, 1)}
                      ]
        self.config = config
        self.act_name = act_name
        self.init_conv2d_layer()

    def init_conv2d_layer(self):
        Conv2dLayer_list = []
        for cfg in self.config:
            conv_layer = torch.nn.Conv2d(in_channels=cfg["in_channels"],
                                         groups=1,
                                         out_channels=cfg["out_channels"],
                                         kernel_size=cfg["kernel_size"],
                                         stride=cfg["stride"],
                                         padding=cfg["padding"],
                                         dilation=(1, 1)
                                         )
            Conv2dLayer_list.append(conv_layer)
            if self.act_name == "relu":
                Conv2dLayer_list.append(torch.nn.ReLU())
            elif self.act_name == "tanh":
                Conv2dLayer_list.append(torch.nn.Tanh())

        # 按顺序组装小网络层为大网络层
        self.Conv2d_Layer = torch.nn.Sequential(
            *Conv2dLayer_list)

    def forward(self, input):
        output = self.Conv2d_Layer(input)
        return output


class Conv2d_Decoder(torch.nn.Module):
    def __init__(self, config=None,
                 act_name="relu"
                 ):
        super(Conv2d_Decoder, self).__init__()
        self.Conv2d_Layer = None
        if config is None:
            config = [
                {"in_channels": 128,
                 "out_channels": 64,
                 "kernel_size": (3, 3),
                 "stride": (1, 1),
                 "padding": (1, 1)},
                {"in_channels": 64,
                 "out_channels": 32,
                 "kernel_size": (3, 3),
                 "stride": (1, 1),
                 "padding": (1, 1)},
                {"in_channels": 32,
                 "out_channels": 16,
                 "kernel_size": (3, 3),
                 "stride": (1, 1),
                 "padding": (1, 1)},
                {"in_channels": 16,
                 "out_channels": 8,
                 "kernel_size": (3, 3),
                 "stride": (1, 1),
                 "padding": (1, 1)},
                {"in_channels": 8,
                 #  "out_channels": 3,  # 输出为三通道即为三方向速度
                 "out_channels": 1,
                 "kernel_size": (3, 3),
                 "stride": (1, 1),
                 "padding": (1, 1)}
            ]
        self.config = config
        self.act_name = act_name
        self.init_conv2d_layer()

    def init_conv2d_layer(self):
        Conv2dLayer_list = []
        for cfg in self.config:
            conv_layer = torch.nn.ConvTranspose2d(in_channels=cfg["in_channels"],
                                                  out_channels=cfg["out_channels"],
                                                  kernel_size=cfg["kernel_size"],
                                                  stride=cfg["stride"],
                                                  padding=cfg["padding"],
                                                  dilation=(1, 1))
            Conv2dLayer_list.append(conv_layer)
            if self.act_name == "relu":
                Conv2dLayer_list.append(torch.nn.ReLU())
            elif self.act_name == "tanh":
                Conv2dLayer_list.append(torch.nn.Tanh())
        self.Conv2d_Layer = torch.nn.Sequential(
            *Conv2dLayer_list)

    def forward(self, input):
        output = self.Conv2d_Layer(input)
        return output


class SunYi_CNNModel(torch.nn.Module):  # python 面向对象，类，父类，继承，初始化
    def __init__(self, act_name=ACT_NAME):
        super(SunYi_CNNModel, self).__init__()
        # init CNN-NetWork
        # 初始化模型参数及网络层
        self.act_name = act_name
        self.encoder_layer = Conv2d_Encoder(act_name=act_name)  # CNN 编码器网络； 卷积网络；
        self.decoder_layer_u = Conv2d_Decoder(act_name=act_name)  # CNN 解码器网络； 反卷积网络
        self.decoder_layer_v = Conv2d_Decoder(act_name=act_name)

    def forward(self, input_emb):  # 网络层前向计算  -->  反向梯度更新【torch自动推断反向更新】
        encode_out = self.encoder_layer(input_emb)  # 编码器网络输出
        # print( "encode_out.shape: {} ".format(encode_out.shape) ) # torch.Size([])
        output_u = self.decoder_layer_u(encode_out)  # 解码器网络输出
        output_v = self.decoder_layer_v(encode_out)  # 解码器网络输出

        output = torch.cat([output_u, output_v], dim=1)
        return output  # 最终输出
