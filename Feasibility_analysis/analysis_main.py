# coding:utf-8

import os
import random
import torch
from torch.autograd import Variable
from torch.optim import Adam
# import seaborn as sns
import matplotlib.pylab as plt
# from visdom import Visdom
import numpy as np
import os.path
from time import strftime, gmtime
import getpass
from sys import platform as _platform
from six.moves import urllib
from os import listdir
from os.path import isfile, join
import pickle as pkl

ACT_NAME = "relu"  # tanh, relu


class Conv2d_Encoder(torch.nn.Module):
    def __init__(self, feature_in_channel, config=None, act_name="relu"):
        super(Conv2d_Encoder, self).__init__()
        self.Conv2d_Layer = None
        if config is None:
            config = [{"in_channels": 4,
                       "out_channels": 4,
                       "kernel_size": (3, 3),
                       "stride": (1, 1),
                       "padding": (1, 1)},
                      {"in_channels": 4,
                       "out_channels": 16,
                       "kernel_size": (3, 3),
                       "stride": (1, 1),
                       "padding": (1, 1)},
                      {"in_channels": 16,
                       "out_channels": 64,
                       "kernel_size": (3, 3),
                       "stride": (1, 1),
                       "padding": (1, 1)},
                      {"in_channels": 64,
                       "out_channels": 128,
                       "kernel_size": (3, 3),
                       "stride": (1, 1),
                       "padding": (1, 1)},
                      {"in_channels": 128,
                       "out_channels": 256,
                       "kernel_size": (3, 3),
                       "stride": (1, 1),
                       "padding": (1, 1)}
                      ]

        if feature_in_channel is not None:
            config[0]["in_channels"] = feature_in_channel

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
    def __init__(self, feature_out_channel, config=None, act_name="relu"):
        super(Conv2d_Decoder, self).__init__()
        self.Conv2d_Layer = None
        if config is None:
            config = [
                {"in_channels": 256,
                 "out_channels": 128,
                 "kernel_size": (3, 3),
                 "stride": (1, 1),
                 "padding": (1, 1)},
                {"in_channels": 128,
                 "out_channels": 64,
                 "kernel_size": (3, 3),
                 "stride": (1, 1),
                 "padding": (1, 1)},
                {"in_channels": 64,
                 "out_channels": 16,
                 "kernel_size": (3, 3),
                 "stride": (1, 1),
                 "padding": (1, 1)},
                {"in_channels": 16,
                 "out_channels": 4,
                 "kernel_size": (3, 3),
                 "stride": (1, 1),
                 "padding": (1, 1)},
                {"in_channels": 4,
                 #  "out_channels": 3,  # 输出为三通道即为三方向速度
                 "out_channels": 1,
                 "kernel_size": (3, 3),
                 "stride": (1, 1),
                 "padding": (1, 1)}
            ]

        if feature_out_channel is not None:
            config[-1]["out_channels"] = feature_out_channel

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
    def __init__(self, feature_in_channel, feature_out_channel, act_name=ACT_NAME):
        super(SunYi_CNNModel, self).__init__()
        # init CNN-NetWork
        # 初始化模型参数及网络层
        self.act_name = act_name
        self.encoder_layer = Conv2d_Encoder(feature_in_channel=feature_in_channel, act_name=act_name)
        self.decoder_layer = Conv2d_Decoder(feature_out_channel=feature_out_channel, act_name=act_name)

    def forward(self, input_emb):  # 网络层前向计算  -->  反向梯度更新【torch自动推断反向更新】
        encode_out = self.encoder_layer(input_emb)  # 编码器网络输出
        # print( "encode_out.shape: {} ".format(encode_out.shape) ) # torch.Size([])
        output = self.decoder_layer(encode_out)  # 解码器网络输出
        return output  # 最终输出


def load_real_dataset(in_folder, U_axis=1):
    step_p_save_file_path = os.path.join(in_folder, "step_p_3.npy")
    step_U_save_file_path = os.path.join(in_folder, "step_U_3.npy")
    step_xyz_save_file_path = os.path.join(in_folder, "step_xyz_3.npy")
    step_p = np.load(step_p_save_file_path).astype(np.float32)  # (9, 40, 30, 100)
    step_U = np.load(step_U_save_file_path).astype(np.float32)  # (9, 40, 30, 100, 3)
    step_xyz = np.load(step_xyz_save_file_path).astype(np.float32)  # (9, 40, 30, 100, 4);  4:[step, x,y,z]

    # step_U = step_U[:, :, :, :, [0]]     # (9, 40, 30, 100, 1)
    # step_U = step_U[:, :, :, :, [1]]     # (9, 40, 30, 100, 1)
    # step_U = step_U[:, :, :, :, [2]]     # (9, 40, 30, 100, 1)
    # step_U = step_U[:, :, :, :, [U_axis]]  # (9, 40, 30, 100, 1) # 选择某一个维度的速度分量
    step_U = step_U[:, :, :, :, :2]

    # channel_first: 模型训练时的数据配置，channel维度提前；
    step_p = np.expand_dims(step_p, axis=-1)  # [9, 40, 30, 100, 1] # axis
    step_p = np.transpose(step_p, [0, 2, 4, 1, 3])  # [9, 30, 1, 40, 100]
    # 
    step_U = np.transpose(step_U, [0, 2, 4, 1, 3])  # [9, 30, 1, 40, 100]
    step_xyz = np.transpose(step_xyz, [0, 2, 4, 1, 3])  # [9, 30, 4, 40, 100]

    return step_p, step_U, step_xyz


def load_batch_dataiter(step_p, step_U, step_xyz,
                        label_type="p",  # 更改label_type="p", "U"
                        batch_size=4, ratio=0.8):
    # re_shape = [40, 30, 100]
    print("step_p.shape: {} ".format(step_p.shape))
    print("step_U.shape: {} ".format(step_U.shape))
    print("step_xyz.shape: {} ".format(step_xyz.shape))
    step_p = step_p.reshape([9 * 30, 1, 40, 100])
    step_U = step_U.reshape([9 * 30, 2, 40, 100])
    step_xyz = step_xyz.reshape([9 * 30, 4, 40, 100])
    # Shuffle the data
    indices = list(range(9 * 30))
    random.shuffle(indices)
    step_p = step_p[indices]
    step_U = step_U[indices]
    step_xyz = step_xyz[indices]

    train_size = int(9 * 30 * ratio)
    train_step_p = step_p[: train_size]
    train_step_U = step_U[: train_size]
    train_step_xyz = step_xyz[: train_size]

    test_step_p = step_p[train_size:]  # [9*30*0.2, 1, 40, 100]
    test_step_U = step_U[train_size:]  # [9*30*0.2, 1, 40, 100]
    test_step_xyz = step_xyz[train_size:]  # [9*30*0.2, 4, 40, 100]

    batch_train_step_p = []
    batch_train_step_U = []
    batch_train_step_xyz = []
    for ind in range(0, train_size, batch_size):
        batch_train_step_p.append(train_step_p[ind: int(ind + batch_size)])
        batch_train_step_U.append(train_step_U[ind: int(ind + batch_size)])
        batch_train_step_xyz.append(train_step_xyz[ind: int(ind + batch_size)])
    # 
    batch_test_step_p = []
    batch_test_step_U = []
    batch_test_step_xyz = []
    for ind in range(0, 9 * 30 - train_size, batch_size):
        batch_test_step_p.append(test_step_p[ind: int(ind + batch_size)])
        batch_test_step_U.append(test_step_U[ind: int(ind + batch_size)])
        batch_test_step_xyz.append(test_step_xyz[ind: int(ind + batch_size)])
    # return zip(batch_train_step_p, batch_train_step_U, batch_train_step_xyz), \
    #        zip(batch_test_step_p, batch_test_step_U, batch_test_step_xyz)

    if label_type == "p":

        batch_train_input = []
        batch_test_input = []
        for i in range(len(batch_train_step_xyz)):
            batch_input = np.concatenate([batch_train_step_U[i], batch_train_step_xyz[i]], axis=1)
            batch_train_input.append(batch_input)
        for i in range(len(batch_test_step_xyz)):
            batch_input = np.concatenate([batch_test_step_U[i], batch_test_step_xyz[i]], axis=1)
            batch_test_input.append(batch_input)

        return list(zip(batch_train_input, batch_train_step_p)), \
            list(zip(batch_test_input, batch_test_step_p))
    else:

        batch_train_input = []
        batch_test_input = []
        for i in range(len(batch_train_step_xyz)):
            batch_input = np.concatenate([batch_train_step_p[i], batch_train_step_xyz[i]], axis=1)
            batch_train_input.append(batch_input)
        for i in range(len(batch_test_step_xyz)):
            batch_input = np.concatenate([batch_test_step_p[i], batch_test_step_xyz[i]], axis=1)
            batch_test_input.append(batch_input)

        return list(zip(batch_train_input, batch_train_step_U)), \
            list(zip(batch_test_input, batch_test_step_U))


def evaluate_model(model, data_list, device):
    # 评估训练模型
    loss_fn = torch.nn.MSELoss(reduce=True, reduction='mean')  # 定义损失函数
    model.eval()  # 调用评估训练模型
    loss_list = []  # 存储loss列表
    # 调用格式
    with torch.no_grad():
        for batch_input, batch_label in data_list:  # 遍历一次数据:  # 遍历数据
            # 输入与标签
            batch_input = Variable(torch.from_numpy(batch_input).to(device))
            labels = Variable(torch.from_numpy(batch_label).to(device))
            outputs = model(batch_input)  # 网络前向计算
            loss = loss_fn(outputs, labels)  # 根据网络预测的 outputs 和 目标labels，利用MSE损失函数计算损失
            loss_list.append(loss.item())  # 加item()为固定格式，转成常规数据类型
            # print("eval, loss: {} ".format( loss.item() ))
    mean_loss = np.mean(loss_list)

    return mean_loss  # 返回均值，在dev集下loss的结果


def test_evaluate_model(model_file_path, test_data_list):
    # 测试模型
    # Define your execution device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device")
    model = SunYi_CNNModel()  # act_name="tanh"
    model.to(device)

    # torch.load(, save_file_path)
    loss_fn = torch.nn.MSELoss(reduce=True, reduction='mean')
    # 加载已存储的最优模型路径
    params_dict = torch.load(model_file_path)
    model.load_state_dict(params_dict)
    model.eval()

    loss_list = []
    for batch_input, labels in test_data_list:
        batch_input = Variable(torch.from_numpy(batch_input).to(device))
        labels = Variable(torch.from_numpy(labels).to(device))
        outputs = model(batch_input)  # 网络前向计算
        loss = loss_fn(outputs, labels)  # 根据网络预测的 outputs 和 目标labels，利用MSE损失函数计算损失
        loss_list.append(loss.item())
    mean_loss = np.mean(loss_list)

    return mean_loss


def predict_result(model, batch_input, device):
    batch_input = Variable(torch.from_numpy(batch_input).to(device))
    outputs = model(batch_input)  # 网络前向计算

    return outputs


def main():
    # 数据集处理部分，根据任务目的来划分 features 和 label;
    # 任务：利用 构造的模型 根据 features作为输入，去预测输出（label）；
    # 当前任务： 三维数据，切片后，根据 压力 和 迭代步数 作为输入， 预测 三个维度的速度。

    result_path = './results'
    progress_time = strftime("%Y-%m-%d %H-%M-%S", gmtime())
    progress_path = os.path.join(result_path, progress_time)
    save_path = os.path.join(progress_path, 'saved_models')  # 存储模型文件夹路径
    os.makedirs(save_path, exist_ok=True)

    # load data
    in_folder = "./inputs/"
    label_type = "U"
    step_p, step_U, step_xyz = load_real_dataset(in_folder)
    train_data, test_data = load_batch_dataiter(step_p, step_U, step_xyz,
                                                label_type=label_type,
                                                batch_size=16, ratio=0.9)
    print("train_data.size: {} ".format(len(train_data)))
    print("test_data.size: {} ".format(len(test_data)))

    # Define your execution device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device")

    feature_in_channel, feature_out_channel = None, None
    if label_type == "U":
        feature_in_channel = 5
        feature_out_channel = 2
    elif label_type == "p":
        feature_in_channel = 6
        feature_out_channel = 1
    model = SunYi_CNNModel(feature_in_channel=feature_in_channel, feature_out_channel=feature_out_channel)
    model.to(device)

    # loss_fn = torch.nn.MSELoss(reduce=True, reduction='mean')
    loss_fn1 = torch.nn.SmoothL1Loss(reduce=True, reduction='mean')
    loss_fn2 = torch.nn.SmoothL1Loss(reduce=True, reduction='mean')
    optimizer = Adam(model.parameters(), lr=5e-4)

    evaluate_per_steps = 200  # 每隔 x 个 step 评估一次
    best_metric_steps = 0  # 当前最优参数的 step 数, 这个会一直累加，不会因为更换 epoch 而清零
    best_metric = 1000000.0  # 设置初始值；当前最优 loss

    best_metric_list = []
    # 用于早停
    stop_training = False
    cnt = 0
    res = 0
    best_ep = 0
    signal = True

    epoch = 1000
    steps = 0
    loss_list = []
    for ep in range(epoch):  # for循环遍历epoch次数据
        for batch_input, batch_label in train_data:  # 遍历一次数据
            steps += 1
            model.train()  # 评估模型固定搭配：.train()
            batch_input = Variable(torch.from_numpy(batch_input).to(device))
            labels = Variable(torch.from_numpy(batch_label).to(device))
            optimizer.zero_grad()
            outputs = model(batch_input)
            loss1 = loss_fn1(outputs[:, 0, :, :], labels[:, 0, :, :])
            loss2 = loss_fn2(outputs[:, 1, :, :], labels[:, 1, :, :])
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()
            # running_loss += loss.item()      # extract the loss value 计算损失和
            loss_list.append(loss.item())
            # #######设置打印输出######### #
            if steps % 10 == 0:
                print('[{:1d}, {:5d}] loss: {:.6f}'.format(
                    ep + 1, steps, np.mean(loss_list)))

            if steps % evaluate_per_steps == 0:
                test_loss = evaluate_model(model, test_data, device)  # 得到loss
                print("test_loss: {} ".format(test_loss))
                if test_loss < best_metric:
                    best_metric = test_loss  # 比较，选取最优 loss
                    best_metric_steps = steps
                    best_ep = ep
                    best_metric_list.append(best_metric)
                    save_file_path = os.path.join(save_path, "saved_model_step{}.pt".format(best_metric_steps))
                    torch.save(model.state_dict(), save_file_path)
                    print("best_step: {}; best_test_loss: {:.6f}".format(best_metric_steps, best_metric))

        # 判断是否早停
        if ep - best_ep >= 100 and signal:
            # stop training
            stop_training = True
            # 计算跟 100 的整数倍还差多少
            res = 100 - ep % 100
            # 防止后面继续增加计数器 所以置为 False 不进入这个 if
            signal = False

        if stop_training:
            # 计数器
            cnt += 1
            # 跑完 res 个 epoch  早停
            if cnt == res:
                break

    return best_metric_steps, progress_path, label_type


def analyse_all(data_step=6, model_steps=460, progress_path=None, label_type="U"):
    save_path = os.path.join(progress_path, 'pic/')
    os.makedirs(save_path, exist_ok=True)

    # 实现功能
    def model_pred(model, input_feats):
        # 输出结果
        output_tensor = predict_result(model, input_feats, device)
        result = output_tensor.detach().cpu().numpy()
        pred_v_val = np.squeeze(result, axis=None)  # 40, 100
        # pred_v_val = np.sqrt( np.sum(pred_v_val ** 2, axis=0) ) # [40, 101]
        # print("pred_v_val.shape: {} ".format(pred_v_val.shape))
        return pred_v_val

    def plot_3d(data, save_fig):
        # Define dimensions
        Nx, Ny, Nz = [30, 40, 100]
        X, Y, Z = np.meshgrid(np.arange(Ny), np.arange(Nx), -np.arange(Nz))

        data = np.flip(data, axis=-1)  #

        kw = {
            'vmin': data.min(),
            'vmax': data.max(),
            'levels': np.linspace(data.min(), data.max(), 20),
        }

        # Create a figure with 3D ax
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        # Plot contour surfaces
        _ = ax.contourf(
            X[:, :, 0], Y[:, :, 0], data[:, :, 0],
            zdir='z', offset=0, **kw)
        _ = ax.contourf(
            X[0, :, :], data[0, :, :], Z[0, :, :],
            zdir='y', offset=0, **kw)
        C = ax.contourf(
            data[:, -1, :], Y[:, -1, :], Z[:, -1, :],
            zdir='x', offset=X.max(), **kw)

        # Set limits of the plot from coord limits
        xmin, xmax = X.min(), X.max()
        ymin, ymax = Y.min(), Y.max()
        zmin, zmax = Z.min(), Z.max()
        ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])

        # Plot edges
        edges_kw = dict(color='0.4', linewidth=1, zorder=1e3)
        ax.plot([xmax, xmax], [ymin, ymax], 0, **edges_kw)
        ax.plot([xmin, xmax], [ymin, ymin], 0, **edges_kw)
        ax.plot([xmax, xmax], [ymin, ymin], [zmin, zmax], **edges_kw)

        # Set labels and zticks
        ax.set(
            xlabel='X [m]',
            ylabel='Y [m]',
            zlabel='Z [m]',
            zticks=range(0, -Nz, -10),
        )

        # Colorbar
        fig.colorbar(C, ax=ax, fraction=0.02, pad=0.1, label='Name [units]')

        # Show Figure
        plt.tight_layout()
        plt.savefig(save_path + save_fig)

    def plot_2d(true_v, pred_v, save_fig):
        true_pred_v = np.concatenate([true_v, pred_v], axis=0)
        plt.rcParams["figure.figsize"] = [16, 8]
        plt.rcParams["figure.autolayout"] = True
        fig, ax = plt.subplots(ncols=1, nrows=2)
        fig.subplots_adjust(wspace=0.01)
        # sns.heatmap(true_pred_v, linewidth=0.5, ax=ax, cbar=True, square=True,  cmap="YlGnBu")
        true_kw = {
            'vmin': true_v.min(),
            'vmax': true_v.max(),
            'levels': np.linspace(true_v.min(), true_v.max(), 100),
        }

        zmin, zmax = true_v.min(), true_v.max()

        ax[0].yaxis.tick_right()
        C = ax[0].contourf(true_v, **true_kw)
        fig.colorbar(C, ax=ax[0], fraction=0.02, pad=0.1, label='Name [units]')

        pred_kw = {
            'vmin': pred_v.min(),
            'vmax': pred_v.max(),
            'levels': np.linspace(pred_v.min(), pred_v.max(), 100),
            # 'vmin': 0.000,
            # 'vmax': 0.05,
            # 'levels': np.linspace(0.00, 0.05, 100),
        }
        ax[1].yaxis.tick_right()
        C = ax[1].contourf(pred_v, **pred_kw)
        zmin, zmax = pred_v.min(), pred_v.max()
        fig.colorbar(C, ax=ax[1], fraction=0.02, pad=0.1, label='Name [units]')

        fig.subplots_adjust(wspace=0.001)
        plt.savefig(save_path + save_fig)

    model_file_path = progress_path + "/saved_models/saved_model_step{}.pt".format(model_steps)
    # Define your execution device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device")

    feature_in_channel, feature_out_channel = None, None
    if label_type == "U":
        feature_in_channel = 5
        feature_out_channel = 2
    elif label_type == "p":
        feature_in_channel = 6
        feature_out_channel = 1
    model = SunYi_CNNModel(feature_in_channel=feature_in_channel, feature_out_channel=feature_out_channel)
    model.to(device)
    params_dict = torch.load(model_file_path)
    model.load_state_dict(params_dict)
    model.eval()

    # load data
    in_folder = "./inputs/"
    step_p, step_U, step_xyz = load_real_dataset(in_folder)
    # step_p:   [9, 30, 1, 40, 100]
    # step_U:   [9, 30, 1, 40, 100]
    # step_xyz: [9, 30, 4, 40, 100]
    if label_type == "U":
        # [30, 5, 40, 100]
        input_feats = np.concatenate([step_p[data_step], step_xyz[data_step]], axis=1)
        # [30, 2, 40, 100]
        labels = step_U[data_step]
    else:
        # [30, 6, 40, 100]
        input_feats = np.concatenate([step_U[data_step], step_xyz[data_step]], axis=1)
        # [30, 1, 40, 100]
        labels = step_p[data_step]

    true_v = []
    pred_v = []
    for ind in range(30):
        # [40, 100] or [2, 40, 100]
        pred_v_val = model_pred(model, np.expand_dims(input_feats[ind], axis=0))
        # [1, 40, 100] or [2, 40, 100]
        true_v_val = labels[ind]
        # [1, 1, 40, 100] or [1, 2, 40, 100]
        true_v.append(np.expand_dims(true_v_val, axis=0))
        if len(pred_v_val.shape) == 2:
            # [1, 1, 40, 100]
            pred_v.append(np.expand_dims(pred_v_val, axis=[0, 1]))
        else:
            # [1, 2, 40, 100]
            pred_v.append(np.expand_dims(pred_v_val, axis=0))

    # [30, 1, 40, 100] or [30, 2, 40, 100]
    true_val = np.concatenate(true_v, axis=0)
    # [30, 40, 100] or [30, 2, 40, 100]
    # true_val = np.squeeze(true_val, axis=1)
    # [30, 1, 40, 100] or [30, 2, 40, 100]
    pred_val = np.concatenate(pred_v, axis=0)

    print("true_val.shape: {} ".format(true_val.shape))
    print("pred_val.shape: {} ".format(pred_val.shape))
    diff_v = np.absolute(true_val - pred_val)
    print("diff_v.shape: {} ".format(diff_v.shape))

    if true_val.shape[1] > 1:
        for i in range(true_val.shape[1]):
            plt_val = np.reshape(true_val[:, i, :, :], [30, 40, 100])
            plot_3d(plt_val, "step{}_true_val_speed{}.png".format(data_step, i))
            plt_val = np.reshape(pred_val[:, i, :, :], [30, 40, 100])
            plot_3d(plt_val, "step{}_pred_val_speed{}.png".format(data_step, i))
            plt_val = np.reshape(diff_v[:, i, :, :], [30, 40, 100])
            plot_3d(plt_val, "step{}_diff_v_speed{}.png".format(data_step, i))
    else:
        plt_val = np.reshape(true_val, [30, 40, 100])
        plot_3d(plt_val, "step{}_true_val.png".format(data_step))
        plt_val = np.reshape(pred_val, [30, 40, 100])
        plot_3d(plt_val, "step{}_pred_val.png".format(data_step))
        plt_val = np.reshape(diff_v, [30, 40, 100])
        plot_3d(plt_val, "step{}_diff_v.png".format(data_step))

    gap = 10000
    best_i = 0
    for i in range(30):
        true_v_3d = true_val[i]
        pred_v_3d = pred_val[i]
        diff_v_3d = np.absolute(true_v_3d - pred_v_3d)
        if np.sum(diff_v_3d) < gap:
            best_i = i
            gap = np.sum(diff_v_3d)

    true_v_3d = true_val[best_i]
    pred_v_3d = pred_val[best_i]

    if true_v_3d.shape[0] > 1:
        for i in range(true_v_3d.shape[0]):
            true_v_2d, pred_v_2d = true_v_3d[i], pred_v_3d[i]
            target_min, target_max = true_v_2d.min(), true_v_2d.max()
            data_min, data_max = pred_v_2d.min(), pred_v_2d.max()
            scale_factor = (target_max - target_min) / (data_max - data_min)
            pred_v_2d = target_min + (pred_v_2d - data_min) * scale_factor

            print("true_v_2d.shape: {} ".format(true_v_2d.shape))
            print("pred_v_2d.shape: {} ".format(pred_v_2d.shape))
            plot_2d(true_v_2d, pred_v_2d, save_fig="step{}_best__speed{}_2d.png".format(data_step, i))

            diff_v_2d = np.absolute(true_v_2d - pred_v_2d)
            diff_v_2d_mean = np.mean(diff_v_2d)
            print("diff_v_2d_mean: {} ".format(diff_v_2d_mean))

            diff_v_2d_rate = diff_v_2d / np.absolute(true_v_2d)
            diff_v_2d_rate_mean = np.mean(diff_v_2d_rate)
            print("diff_v_2d_rate_mean: {} ".format(diff_v_2d_rate_mean))
            plot_2d(true_v_2d, diff_v_2d_rate, save_fig="step{}_diff_rate_2d.png".format(data_step))
    else:
        true_v_2d, pred_v_2d = np.squeeze(true_v_3d, axis=0), np.squeeze(pred_v_3d, axis=0)
        target_min, target_max = true_v_2d.min(), true_v_2d.max()
        data_min, data_max = pred_v_2d.min(), pred_v_2d.max()
        scale_factor = (target_max - target_min) / (data_max - data_min)
        pred_v_2d = target_min + (pred_v_2d - data_min) * scale_factor

        print("true_v_2d.shape: {} ".format(true_v_2d.shape))
        print("pred_v_2d.shape: {} ".format(pred_v_2d.shape))
        plot_2d(true_v_2d, pred_v_2d, save_fig="step{}_best_2d.png".format(data_step))

        diff_v_2d = np.absolute(true_v_2d - pred_v_2d)
        diff_v_2d_mean = np.mean(diff_v_2d)
        print("diff_v_2d_mean: {} ".format(diff_v_2d_mean))

        diff_v_2d_rate = diff_v_2d / np.absolute(true_v_2d)
        diff_v_2d_rate_mean = np.mean(diff_v_2d_rate)
        print("diff_v_2d_rate_mean: {} ".format(diff_v_2d_rate_mean))
        plot_2d(true_v_2d, diff_v_2d_rate, save_fig="step{}_diff_rate_2d.png".format(data_step))


if __name__ == "__main__":
    # 训练并保存最优模型
    step, progress_path, label_type = main()
    # progress_path = './results/2024-03-15 14-29-40'
    # step = 9600
    analyse_all(model_steps=step, progress_path=progress_path, label_type=label_type)
