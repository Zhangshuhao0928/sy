#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/3/20 21:34
# @Author : Erhaoo
# @Belief : Everything will be fine～

import os
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter

# 路径
path = "./UNET"
dir_path = os.path.join(path, '10-7-650-no')
os.makedirs(dir_path, exist_ok=True)
data_path = os.path.join(path, 'curves.csv')

# 加载数据
data = pd.read_csv(data_path, header=0)

# 列名帮你写出来了，根据下标索引
col = ['Epoch',
       'Train Loss',
       'Train Total MSE',
       'Train Ux MSE',
       'Train Uy MSE',
       'Val Loss',
       'Val Total MSE',
       'Val Ux MSE',
       'Val Uy MSE',
       'Test Loss',
       'Test Total MSE',
       'Test Ux MSE',
       'Test Uy MSE',
       'r_squre']

for i in range(len(col)):
    COL = col[i]
    # 把 loss 拿出来
    loss = data[COL].tolist()

    # 越大曲线越平滑，只能是奇数，不可以是偶数
    window_len = 81
    # 越小曲线越平滑，1就是最小， 且一定要先烤鱼 window_len
    polyorder = 1

    # 平滑函数  loss 是数据  window_len 是窗口长度
    # loss_smooth = savgol_filter(loss, window_len, polyorder, mode='nearest')

    plt.figure(figsize=(10, 7), dpi=650)

    plt.plot(loss, color='cornflowerblue', linewidth=1.5, label=COL)

    # 罗马字体
    plt.rcParams['font.family'] = 'Times New Roman'
    # # 小五号 就是 10
    plt.rcParams['font.size'] = 28


    # # 下面这几行代码是用于控制是否添加 times roman 的
    plt.xlabel('Epoch')
    y_label = '_'.join(COL.split(' '))
    plt.ylabel(y_label)

    # plt.legend()
    plt.title(COL)

    plt.tight_layout()
    if i != 0:
        save_path = os.path.join(dir_path, COL)
        plt.savefig(save_path + '.png')
        plt.show()

# 爱心发射❤️，biu～ 哈哈哈哈哈哈哈 啦啦啦
