#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/2/21 13:26
# @Author : Erhaoo
# @Belief : Everything will be fine～

# coding:utf-8

import os
import random
import numpy as np
import sys
import matplotlib.pyplot as plt
from typing import List

# 用于控制 print 语句输出的小数的显示精度； 参数 sys.maxsize 表示无论有多少位的小数，都全精度输出
np.set_printoptions(threshold=sys.maxsize)


def load_theta_file(file_path: str) -> np.array:
    """
    @param file_path: string format
    @return: array which only contains all vals but not steps
    """

    data_list = []
    with open(file_path, "r", encoding="utf-8") as f_in:
        for line in f_in:
            # 去除每一行两端多余的换行符
            line = line.strip("\n")
            # 按照 空格符 进行切分 （这里的数据每一行会切分为 2 个，第一个是 index，第二个是值，也就是等差数列应用的地方）
            items = line.split(" ")
            # 过滤无效数据，转换成 list
            items = list(filter(lambda x: x != "", items))
            # 判断 items 长度是否为 2，因为上面说过了每一行会切分为 2 个，不是 2 则说明有问题，这种数据略过不要
            if len(items) != 2:
                continue
            # 分别取出 index（step）以及值（val）
            step, val = items
            # 只添加值，step 这里没有使用
            data_list.append(val)
    # 将 list 转换成 numpy 的数组形式，数据格式为 32位浮点数
    data_array = np.asarray(data_list, dtype=np.float32)

    return data_array


def read_sdf_file(file_path: str) -> np.array:
    """
    @param file_path: sdf file path
    @return: array
    """

    data_list = []
    with open(file_path, "r", encoding="utf-8") as f_in:
        line_array = []
        for line in f_in:
            # 数据中每行开头会有 4 个空格，要去除掉
            line = line.strip("\n").strip(" ")
            # 切割数据
            items = line.split(" ")
            # 过滤无效数据
            items = list(filter(lambda x: x != "", items))
            # 防止数据中有 int 类型的数值，将全部数据转换成 float 类型并转换成 list 对象
            # 这里的 extend 方法是逐个添加，而非整体添加，区别于 append 方法
            line_array.extend(list(map(float, items)))
            # 每经过 62 行后，会出现一行只有 2 个数据（62x4+2=250），所以这里是判断一条数据是否完整
            # 当到了 250维 之后则开始整个 append 操作到 data_list 中
            # 添加完之后，清空 line_array 用于继续添加下一条数据
            if len(items) < 4:
                data_list.append(line_array)
                line_array = []
    # 转换为数组
    data_array = np.asarray(data_list, dtype=np.float32)  # (50, 250)
    return data_array


# 制作数据集主体函数
def process() -> None:
    input_feats_list = []
    uvArray_list = []

    for init_U in [0.3, 0.4, 0.5, 0.6, 0.7]:
        # 按照等差数列设计的文件，每一行第二列的差值为 0.095
        theta_file_path = "para0_U100_U300.txt"
        # 加载 theta_file
        theta_vals = load_theta_file(theta_file_path)
        print("theta_vals.shape: {} ".format(theta_vals.shape))  # (201,)

        for pre in ["340", "440", "540", "640", "740"]:
            data_path = "./{}/{}/".format(str(init_U), pre)
            sdf_file_name = ["sdf11.dat", "sdf10.dat"]
            sdf_data_list = []

            for file_name in sdf_file_name:
                # 路径拼接操作，获取 sdf 文件路径
                file_path = os.path.join(data_path, file_name)
                # 读取 sdf 文件
                data_array = read_sdf_file(file_path)  # (50, 250)
                # 在第零个维度扩充数据，将原本的二维扩充到三维
                data_array = np.expand_dims(data_array, axis=0)  # (1, 50, 250)
                # 将 data_array 整个append 到 sdf_data_list 中，由于前面我们只使用了 sdf11 和 sdf10  这两个 dat
                # 所以最终 sdf_data_list 中有两个 shape 为（1, 50, 250）的 list 对象
                sdf_data_list.append(data_array)
            # 在第 1 个维度拼接，也就是将两个（1, 50, 250）拼接成 （2,50 ,250）
            sdf_array = np.concatenate(sdf_data_list, axis=0)  # (2, 50, 250)
            # print("sdf_array.shape: {} ".format(sdf_array.shape))  # sdf_array.shape: (2, 50, 250)

            # create Init_U and theta_val as init_feats.
            init_u_arr = np.asarray([init_U], dtype=np.float32)  # (1,)
            # 扩展维度，将原本的一个数值扩充为 shape 为（1,50 ,250）的数组，数组内的每一个数值都是一模一样的
            # 例如 init_u 为 0.3 时，数组中全部的数值都是 0.3
            init_u_arr = np.tile(init_u_arr, [1, 50, 250])  # (1, 50, 250)
            init_feats_list = []
            # theta_vals 为等差数列的那 201 个值
            for theta in theta_vals:
                # 转换为数组
                tmp_theta_arr = np.asarray([theta], dtype=np.float32)  # (1,)
                # 扩充维度，数组内的数值都是一模一样的
                tmp_theta_arr = np.tile(tmp_theta_arr, [1, 50, 250])  # (1, 50, 250)
                # 在第一个维度拼接
                init_feats_array = np.concatenate([init_u_arr, tmp_theta_arr], axis=0)  # (2, 50, 250)
                # 增加第 0 个维度，从三维变成了四维
                init_feats_array = np.expand_dims(init_feats_array, axis=0)  # (1, 2, 50, 250)
                init_feats_list.append(init_feats_array)

            # 把这 201 个 list 在第 1 个维度拼接
            # 得到的 init_feats_array 后两个维度（50和 250）都是一模一样的数字
            init_feats_array = np.concatenate(init_feats_list, axis=0)  # (201, 2, 50 ,250)
            # print("init_feats_array.shape: {} ".format( init_feats_array.shape ))      # shape:(201, 2, 50, 250)

            # 扩充维度，将原本的(2, 50, 250) 扩充为 (201, 2, 50, 250) 这里可以理解为复制，即 201个三维数组是一模一样的
            input_sdf_feats = np.tile(sdf_array, [201, 1, 1, 1])

            selected_feats = [input_sdf_feats, init_feats_array]
            # selected_feats = [input_sdf_feats]
            # 在第一个维度进行拼接 数据的形式是：原本的两个 sdf 的数据 + init_u + 等差数列中的数（+的意思是拼接 不是加法）
            input_feats = np.concatenate(selected_feats, axis=1)  # (201, 4, 50, 250)
            # print("input_feats.shape: {} ".format( input_feats.shape )) # input_feats.shape: (201, 4, 50, 250)

            # 这两个 list 对象中的每个元素都是一个 string 字符串 形式如["U100.dat", "U101.dat", ... "U300.dat"]
            U_file_name = ["U{}.dat".format(i) for i in range(100, 301, 1)]
            V_file_name = ["V{}.dat".format(i) for i in range(100, 301, 1)]

            UV_data_list = []
            # zip是打包操作，每次分别从 U_file_name, V_file_name中取出下标对应的 U，V 文件，例如 U100.dat 和 V100.dat
            # 循环会遍历 201 次
            for u_f, v_f in zip(U_file_name, V_file_name):
                U_file_path = os.path.join(data_path, u_f)
                V_file_path = os.path.join(data_path, v_f)
                U_array = read_sdf_file(U_file_path)  # (50, 250)
                V_array = read_sdf_file(V_file_path)  # (50, 250)
                # 扩充维度
                U_array = np.expand_dims(U_array, axis=0)  # (1, 50, 250)
                V_array = np.expand_dims(V_array, axis=0)  # (1, 50, 250)
                # print(U_file_path,U_array.shape,V_array.shape)
                # 第一个维度拼接
                UV_array = np.concatenate([U_array, V_array], axis=0)  # (2, 50, 250)
                # 扩充维度
                UV_array = np.expand_dims(UV_array, axis=0)  # (1, 2, 50, 250)
                UV_data_list.append(UV_array)
            # 第一个维度拼接
            All_UV_array = np.concatenate(UV_data_list, axis=0)  # (201, 2, 50, 250)
            # print("All_UV_array.shape: {} ".format( All_UV_array.shape ))  # All_UV_array.shape: (201, 2, 50, 250)

            # 这两个 list 对象会执行 25 次 append 操作，因为 init_u 有 5 个，每个 init_u 中又分 340，440... 5个
            # 共 25 次循环
            input_feats_list.append(input_feats)
            uvArray_list.append(All_UV_array)

    # 第一个维度拼接
    all_input_feats = np.concatenate(input_feats_list, axis=0)  # (5025, 4, 50, 250)
    all_uv_labels = np.concatenate(uvArray_list, axis=0)  # (5025, 2, 50, 250)
    print("all_input_feats.shape: {} ".format(all_input_feats.shape))
    print("all_uv_labels.shape: {} ".format(all_uv_labels.shape))

    save_path = "./inputs/"
    os.makedirs(save_path, exist_ok=True)
    # np.save(os.path.join(save_path, "input_feats.npy"), all_input_feats)
    # np.save(os.path.join(save_path, "input_uv_labels.npy"), all_uv_labels)

    data_x = all_input_feats
    data_y = all_uv_labels
    data_x = data_x.astype(np.float32)
    data_y = data_y.astype(np.float32)

    # # min-max scaler  归一化操作 将所有的数都放缩到 0～1 之间，之前的原始数据中会有负数
    # data_x_max, data_x_min, data_y_max, data_y_min = data_x.max(), data_x.min(), data_y.max(), data_y.min()
    # data_x = (data_x - data_x_min) / (data_x_max - data_x_min)
    # data_y = (data_y - data_y_min) / (data_y_max - data_y_min)

    # 0.8 即 1/4 用于制作测试集
    ratio = 0.8
    # index 列表， 为 [0, 1, 2, ... , 5024]
    indices = list(range(data_y.shape[0]))
    # 设置随机数种子  2024加油  everything will be fine.
    np.random.seed(2024)
    # 将 indices 中的下标打乱顺序
    random.shuffle(indices)
    # 按照打乱后的数据下标进行排列
    data_x = data_x[indices]
    data_y = data_y[indices]

    train_size = int(data_y.shape[0] * ratio)  # 4020
    # 取前 4020 个数据作为训练集，后 1005 个数据作为测试级
    train_x = data_x[:train_size]
    train_y = data_y[:train_size]
    test_x = data_x[train_size:]
    test_y = data_y[train_size:]

    # np.save(os.path.join(save_path, "train_x_shuffle.npy"), train_x)
    # np.save(os.path.join(save_path, "train_y_shuffle.npy"), train_y)
    # np.save(os.path.join(save_path, "test_x_shuffle.npy"), test_x)
    # np.save(os.path.join(save_path, "test_y_shuffle.npy"), test_y)
    print('train_x:', train_x.shape)
    print('train_y:', train_y.shape)
    print('test_x:', test_x.shape)
    print('test_y:', test_y.shape)


# demo 画图函数
def plot_demo(true_v: np.array, save_fig: str, shape: List = [25, 5], min_max_v: List = None) -> None:
    # 设置 plt 基础参数
    plt.rcParams["figure.figsize"] = shape
    plt.rcParams["figure.autolayout"] = True
    # 设置子图，但是这里目前只有 1 个图，其实不用子图也行
    fig, ax = plt.subplots(ncols=1, nrows=1)
    # 调整子图布局，目前就 1 个子图，所以没啥用
    fig.subplots_adjust(wspace=0.01)

    # 画等高线用
    if min_max_v is None:
        true_kw = {
            'vmin': true_v.min(),
            'vmax': true_v.max(),
            'levels': np.linspace(true_v.min(), true_v.max(), 100),
        }
    else:
        true_kw = {
            'vmin': min_max_v[0],
            'vmax': min_max_v[1],
            'levels': np.linspace(min_max_v[0], min_max_v[1], 100),
        }
    # 画等高线
    C = ax.contourf(true_v, extent=[0, 250, 0, 50], **true_kw)
    # colorbar 设置，这里我改了，改完之后更好看点
    fig.colorbar(C, ax=ax, fraction=0.01, pad=0.01, orientation='vertical', label='Name [units]')
    # 无用
    fig.subplots_adjust(wspace=0.001)
    # 保存图片 注意目前的路径是写死的，不是灵活的
    plt.savefig(save_fig)


# 画测试图 sdf U V
def test_plot_data() -> None:
    file_path = "./0.{}/{}40/sdf{}.dat".format("4", "4", "11")
    sdf_data = read_sdf_file(file_path)
    sdf_array = np.asarray(sdf_data, dtype=np.float32)  # (50, 250)
    plot_demo(sdf_array, save_fig="sdf11_pic.png")

    file_path = "./0.{}/{}40/U{}.dat".format("4", "4", 200)
    sdf_data = read_sdf_file(file_path)
    sdf_array = np.asarray(sdf_data, dtype=np.float32)  # (50, 250)
    plot_demo(sdf_array, save_fig="U200_pic.png", min_max_v=[-1.0, 0.02])

    file_path = "./0.{}/{}40/V{}.dat".format("4", "4", 200)
    sdf_data = read_sdf_file(file_path)
    sdf_array = np.asarray(sdf_data, dtype=np.float32)  # (50, 250)
    plot_demo(sdf_array, save_fig="V200_pic.png", min_max_v=[-0.1, 0.01])


if __name__ == "__main__":
    test_plot_data()
    # process()

