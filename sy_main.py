#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/2/21 13:30
# @Author : Erhaoo
# @Belief : Everything will be fine～

import os.path
import csv
import pandas as pd
import sys
import time

from train_functions import *
from functions import *
from torch.utils.data import TensorDataset
from Models.UNetEx import UNetEx
from Models.T_CNN import SunYi_CNNModel
from typing import List, Tuple, Dict
from time import strftime, gmtime


# 指定显卡 如果只有一块显卡的话则默认是 0
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def load_data() -> Tuple:
    load_path = "./data/inputs"
    data_x = np.load(os.path.join(load_path, "train_x_shuffle.npy"))
    data_y = np.load(os.path.join(load_path, "train_y_shuffle.npy"))
    data_x = data_x.astype(np.float32)
    data_y = data_y.astype(np.float32)
    # 将 np.array 转换成 tensor 向量
    data_x = torch.tensor(data_x)
    data_y = torch.tensor(data_y)

    test_x = np.load(os.path.join(load_path, "test_x_shuffle.npy"))
    test_y = np.load(os.path.join(load_path, "test_y_shuffle.npy"))
    test_x = test_x.astype(np.float32)
    test_y = test_y.astype(np.float32)
    test_x = torch.tensor(test_x)
    test_y = torch.tensor(test_y)
    # load_path = "./340"
    # data_x = np.load(os.path.join(load_path, "input_feats.npy"))
    # data_y = np.load(os.path.join(load_path, "input_uv_labels.npy"))
    # data_x = data_x.astype(np.float32)
    # data_y = data_y.astype(np.float32)
    # # 将 np.array 转换成 tensor 向量
    # data_x = torch.tensor(data_x)
    # data_y = torch.tensor(data_y)
    #
    # test_x = np.load(os.path.join(load_path, "input_feats_test.npy"))
    # test_y = np.load(os.path.join(load_path, "input_uv_labels_test.npy"))
    # test_x = test_x.astype(np.float32)
    # test_y = test_y.astype(np.float32)
    # test_x = torch.tensor(test_x)
    # test_y = torch.tensor(test_y)

    return data_x, data_y, test_x, test_y


def main(mode: str = None) -> None:
    """
    @type mode: 模式
    """

    result_path = './results'
    # 引入时间，方便多进程跑程序，每个程序的结果互不干扰，不会覆盖
    progress_time = strftime("%Y-%m-%d %H-%M-%S", gmtime())
    progress_path = os.path.join(result_path, progress_time)
    print('Current progress path is :', progress_path)
    # 创建当前程序的 result 文件夹
    os.makedirs(progress_path, exist_ok=True)

    plot_path = os.path.join(progress_path, "plot")
    os.makedirs(plot_path, exist_ok=True)
    save_model_path = os.path.join(progress_path, "saved_models")
    os.makedirs(save_model_path, exist_ok=True)
    dat_path = os.path.join(progress_path, "dat")
    os.makedirs(dat_path, exist_ok=True)

    all_path = {'progress_path': progress_path, 'plot': plot_path, 'saved_models': save_model_path, 'dat': dat_path}

    # 指定设备，gpu 可用时为 GPU，否则为 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x, y, test_x, test_y = load_data()

    # 创建 PyTorch 的 TensorDataset
    train_dataset_for_print = TensorDataset(x, y)
    train_dataset = (x, y)
    test_dataset = TensorDataset(test_x, test_y)

    # 打印数据形状
    print("Training set size:", len(train_dataset_for_print))
    print("Testing set size:", len(test_dataset))

    # 获取输入数据第二个维度（即通道维度）大小
    dim_1 = x.shape[1]
    # 获取每个通道的 平方均值 后再开根号，最后变换维度为 (1, dim_1, 1, 1)
    channels_weights = torch.sqrt(torch.mean(y.permute(0, 2, 3, 1).reshape((-1, dim_1)) ** 2, dim=0)). \
        view(1, -1, 1, 1).to(device)

    # 设置 CPU 随机数种子，所有的随机数种子最好一致，生成数据使用的是 2024，则这里也用 2024 较好
    torch.manual_seed(2024)
    # 学习率
    lr = 1e-3
    init_lr = 5e-4
    warm_up_iter = 0

    # 卷积核大小，只写一个数则默认卷积核是 5x5
    kernel_size = 5
    # 卷积核个数，决定卷积后的通道维度数
    filters = [8, 16, 32, 32]
    # 批量归一化
    bn = False
    # 权重归一化
    wn = False
    # 权重衰减系数
    wd = 0.003

    # mode：tcnn unet sy
    # mode = 'sy'

    if mode == 'sy':
        def swish(xx):
            return xx * torch.sigmoid(xx)

        class Swish(nn.Module):
            def __init__(self):
                super(Swish, self).__init__()

            def forward(self, xx):
                return swish(xx)

        # 设置网络模型, 之前报错的原因就是这里第一个参数值写死为 2 的原因
        # 这个 bug 已经修复了
        model = UNetEx(dim_1, 2, filters=filters, kernel_size=kernel_size, batch_norm=bn, weight_norm=wn)

        # Define optimizer，这里使用的是 adam 优化器
        optimizer = torch.optim.AdamW(model.parameters(), lr=init_lr, weight_decay=wd)
    elif mode == 'unet':
        model = UNetEx(dim_1, 2, filters=filters, kernel_size=kernel_size, batch_norm=bn, weight_norm=wn)
        optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, weight_decay=wd)
    else:
        model = SunYi_CNNModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, weight_decay=wd)

    # 记录各种值（之前 print 的那一堆）
    train_loss_curve = []
    val_loss_curve = []
    test_loss_curve = []
    train_mse_curve = []
    val_mse_curve = []
    test_mse_curve = []
    train_ux_curve = []
    val_ux_curve = []
    test_ux_curve = []
    train_uy_curve = []
    val_uy_curve = []
    test_uy_curve = []
    r_square = []

    # 训练的总轮次
    epochs = 4000
    # 峰值训练轮次
    normal_epochs = int(1 * epochs)
    # batch 大小
    batch_size = 100
    # 间隔 interval 测试一次模型性能
    test_interval = 200
    # 间隔 interval 画一次模型 curve 图
    plot_interval = 200
    # k-折交叉验证
    kf = 10

    # 内联函数  学习率热启动，防止一开始训练的时候学习率太大造成训练波动不稳定
    def warm_up(current_epoch: int, init_lr: float = init_lr, base_lr: float = lr, warm_up_iter: int = warm_up_iter,
                optimizer=optimizer) -> None:
        """
        @param optimizer: 优化器
        @param current_epoch: 当前训练轮次数
        @param init_lr: 初始学习率
        @param base_lr: 基准学习率
        @param warm_up_iter: 学习率热启动迭代轮次数
        """

        if current_epoch < warm_up_iter:
            alpha = current_epoch / warm_up_iter
            learning_rate = init_lr + (base_lr - init_lr) * (1 - np.exp(-5 * alpha))
        else:
            learning_rate = base_lr

        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

    # 内联函数 学习率衰减
    def lr_decay(current_epoch: int, total_epoch: int = epochs, normal_epochs: int = normal_epochs,
                 optimizer=optimizer, base_lr: float = lr) -> None:
        """
        @param normal_epochs: 峰值训练轮次数
        @param current_epoch: 当前训练轮次数
        @param total_epoch: 总训练轮次
        @param optimizer: 优化器
        @param base_lr: 基准学习率
        """

        gap = current_epoch - normal_epochs
        learning_rate = base_lr - (base_lr * (gap / float(total_epoch)))
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

    # 内联函数，用于测试模型
    def test_model(model: nn.Module, epoch_id: int, test_x: Any = test_x, test_y: Any = test_y, device: Any = device,
                   all_path: Dict = all_path) -> None:
        """
        @param all_path: 全部存储路径
        @param model: 训练之后存储的模型，至于是否用所谓的最优模型有待商榷
        @param test_x: 测试数据
        @param test_y: 测试 label
        @param device: 设备
        @param epoch_id: 当前 epoch 轮次
        """

        model.eval()
        with torch.no_grad():
            out = model(test_x.to(device))
            error = torch.abs(out.cpu() - test_y.cpu())

        # 可视化预测结果
        visualize(test_y.cpu().detach().numpy(), out.cpu().detach().numpy(), error.cpu().detach().numpy(), epoch_id,
                  all_path, test_interval=test_interval)

    # 内联函数，避免传递参数
    def after_epoch(scope: Dict, epoch_id: int = None, plot_interval: int = plot_interval,
                    progress_path: str = progress_path, test_mode: bool = False) -> None:
        """
        @param test_mode: 测试模式
        @param plot_interval: 画图间隔
        @param epoch_id: 当前 epoch 轮次
        @param progress_path: 程序路径
        @param scope: 存储各种损失值以及 mse 值的字典
        """

        train_loss_curve.append(scope["train_loss"])
        val_loss_curve.append(scope["val_loss"])
        train_mse_curve.append(scope["train_metrics"]["mse"])
        val_mse_curve.append(scope["val_metrics"]["mse"])
        train_ux_curve.append(scope["train_metrics"]["ux"])
        val_ux_curve.append(scope["val_metrics"]["ux"])
        train_uy_curve.append(scope["train_metrics"]["uy"])
        val_uy_curve.append(scope["val_metrics"]["uy"])
        test_loss_curve.append(scope["test_loss"])
        test_mse_curve.append(scope["test_metrics"]["mse"])
        test_ux_curve.append(scope["test_metrics"]["ux"])
        test_uy_curve.append(scope["test_metrics"]["uy"])
        r_square.append(scope["r"])

        curves = {
            "train_loss_curve": train_loss_curve, "train_mse_curve": train_mse_curve,
            "train_ux_curve": train_ux_curve, "train_uy_curve": train_uy_curve,
            "val_loss_curve": val_loss_curve, "val_mse_curve": val_mse_curve,
            "val_ux_curve": val_ux_curve, "val_uy_curve": val_uy_curve,
            "test_loss_curve": test_loss_curve, "test_mse_curve": test_mse_curve,
            "test_ux_curve": test_ux_curve, "test_uy_curve": test_uy_curve,
            "r_square": r_square
        }

        if epoch_id % plot_interval == 0:
            plot_curves(curves, progress_path)

    # 损失函数定义，内联函数
    def loss_func(model: nn.Module, batch: torch.Tensor) -> Tuple:
        """
        @param model: 模型
        @param batch: 批量数据
        @return:
            batch 上的总 loss
            模型输出
        """

        # 取出输入特征 x 以及 uv_label y
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        model = model.to(device)
        # 模型的前向计算 forward
        output = model(x)

        beta = 1.0

        # 计算 Ux、Uy 方向上的损失
        # output[:, i, :, :] 表示模型预测输出的第 i 个通道
        # y[:, i, :, :] 表示实际标签的第 i 个通道
        # 输出的 size 是三维，通道维度没了，所以要 reshape 到四维，通道维度设置为 1
        if mode == 'sy':
            # diff_u = torch.abs(output[:, 0, :, :] - y[:, 0, :, :]).reshape(
            #     (output.shape[0], 1, output.shape[2], output.shape[3]))
            # diff_v = torch.abs(output[:, 1, :, :] - y[:, 1, :, :]).reshape(
            #     (output.shape[0], 1, output.shape[2], output.shape[3]))
            # lossu = torch.where(diff_u < beta, 0.5 * diff_u ** 2 / beta, diff_u - 0.5 * beta)
            # lossv = torch.where(diff_v < beta, 0.5 * diff_v ** 2 / beta, diff_v - 0.5 * beta)
            lossu = ((output[:, 0, :, :] - y[:, 0, :, :]) ** 2).reshape(
                (output.shape[0], 1, output.shape[2], output.shape[3]))
            lossv = ((output[:, 1, :, :] - y[:, 1, :, :]) ** 2).reshape(
                (output.shape[0], 1, output.shape[2], output.shape[3]))
        else:
            lossu = ((output[:, 0, :, :] - y[:, 0, :, :]) ** 2).reshape(
                (output.shape[0], 1, output.shape[2], output.shape[3]))
            lossv = ((output[:, 1, :, :] - y[:, 1, :, :]) ** 2).reshape(
                (output.shape[0], 1, output.shape[2], output.shape[3]))

        # 计算损失，这里的损失是整个 batch 上的损失值
        if mode == 'sy':
            loss = lossu + lossv
        else:
            loss = (lossu + lossv) / channels_weights

        # 这里的 sum loss 没有除以样本数目，存疑
        # print(torch.sum(loss),torch.sum(lossu + lossv))
        return torch.sum(loss), output

    # Training
    start_time = time.time()
    best_model, best_train_metrics, best_train_loss, best_test_metrics, best_test_loss, best_epoch_id = train_model(
        model,  # 使用的模型
        loss_func,  # 定义的损失函数
        train_dataset,  # 训练数据集
        test_dataset,  # 测试数据集
        optimizer,  # 优化器
        epochs=epochs,  # 训练的总轮次
        normal_epochs=normal_epochs,  # 峰值训练轮次
        warm_up_iter=warm_up_iter,  # 学习率预热轮次
        batch_size=batch_size,  # 每批次的样本数量
        device=device,  # 使用的设备（GPU 或 CPU）
        # 下面的这些都是 lambda 匿名函数，其参数均为一个字典类型的参数 scope
        # scope["batch"] 是一个 batch 的数据包括输入特征以及label, 所以下标[1]表明是 label，用于计算 mse
        # 每个 epoch 上  Total MSE = Ux MSE + Uy MSE，其中 Total MSE = 整个 epoch 的（lossu+lossv）/训练集数目
        # 这里的 lossu+lossv 不除以 channel_weights
        m_mse_name="Total MSE",  # 总的均方误差指标名称
        m_mse_on_batch=lambda scope: float(torch.sum((scope["output"] - scope["batch"][1]) ** 2)),  # 计算每个批次的均方误差
        m_mse_on_epoch=lambda scope: sum(scope["list"]) / len(scope["dataset"]),  # 计算每个 epoch 的均方误差，这里除以样本数目了
        m_ux_name="Ux MSE",  # Ux 方向的均方误差指标名称
        m_ux_on_batch=lambda scope: float(
            torch.sum((scope["output"][:, 0, :, :] - scope["batch"][1][:, 0, :, :]) ** 2)),  # 计算每个批次的 Ux 方向的均方误差
        m_ux_on_epoch=lambda scope: sum(scope["list"]) / len(scope["dataset"]),  # 计算每个 epoch 的 Ux 方向的均方误差
        m_uy_name="Uy MSE",  # Uy 方向的均方误差指标名称
        m_uy_on_batch=lambda scope: float(
            torch.sum((scope["output"][:, 1, :, :] - scope["batch"][1][:, 1, :, :]) ** 2)),  # 计算每个批次的 Uy 方向的均方误差
        m_uy_on_epoch=lambda scope: sum(scope["list"]) / len(scope["dataset"]),  # 计算每个 epoch 的 Uy 方向的均方误差
        patience=25,  # 用于 early stopping 的耐心值
        after_epoch=after_epoch,  # 每个 epoch 后执行的回调函数
        all_path=all_path,  # 全部用到的存储路径
        test_model=test_model,  # 测试模型函数
        test_interval=test_interval,  # 间隔测试时间
        warm_up=warm_up,  # 学习率预热函数
        lr_decay=lr_decay,  # 学习率衰减函数
        kf=kf,  # k-折交叉验证
        mode=mode
    )

    end_time = time.time()
    total_seconds = end_time - start_time
    epoch_seconds = total_seconds / epochs

    # 将总时间转换为时分秒格式
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)

    epoch_hours = int(epoch_seconds // 3600)
    epoch_minutes = int((epoch_seconds % 3600) // 60)
    epoch_seconds = int(epoch_seconds % 60)

    experiment_duration = {
        "hours": hours,
        "minutes": minutes,
        "seconds": seconds
    }

    epoch_duration = {
        "hours": epoch_hours,
        "minutes": epoch_minutes,
        "seconds": epoch_seconds
    }

    # 打印总时间和转换后的时分秒格式
    print(f"实验总共运行了 {hours} 小时 {minutes} 分钟 {seconds} 秒")
    print(f"每个运行了 {epoch_hours} 小时 {epoch_minutes} 分钟 {epoch_seconds} 秒")

    # After training:
    # 保存全部的最优数据，这个我是看着你原来的结构给你写的，但是实际上这个东西目前没用，先帮你存起来把，看看以后能不能有用
    # 保存最优模型参数
    best_model_save_path = os.path.join(save_model_path, f"best_model_epoch{best_epoch_id}.pt")
    torch.save(best_model.state_dict(), best_model_save_path)
    print(f"最优模型成功保存：{best_model_save_path}")
    # 保存全部最优指标
    best_config = {
        'best_train_metrics': best_train_metrics,
        'best_train_loss': best_train_loss,
        'best_test_metrics': best_test_metrics,
        'best_test_loss': best_test_loss,
        'experiment_time': experiment_duration,
        'epoch_time': epoch_duration
    }
    best_config_save_path = os.path.join(progress_path, "best_config.csv")
    with open(best_config_save_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for row in best_config.items():
            writer.writerow(row)

    # 保存所有 curves 数据
    df_data = {
        'Epoch': list(range(1, len(train_loss_curve) + 1)),
        'Train Loss': train_loss_curve,
        'Train Total MSE': train_mse_curve,
        'Train Ux MSE': train_ux_curve,
        'Train Uy MSE': train_uy_curve,
        'Val Loss': val_loss_curve,
        'Val Total MSE': val_mse_curve,
        'Val Ux MSE': val_ux_curve,
        'Val Uy MSE': val_uy_curve,
        'Test Loss': test_loss_curve,
        'Test Total MSE': test_mse_curve,
        'Test Ux MSE': test_ux_curve,
        'Test Uy MSE': test_uy_curve,
        'r_squre': r_square
    }

    df = pd.DataFrame(df_data).set_index('Epoch')
    df.to_csv(progress_path + '/' + 'curves.csv')
    print("Curves have been saved to: ", progress_path + '/' + 'curves.csv')

    print("Process done !")


if __name__ == "__main__":
    mode = sys.argv[1]
    main(mode=mode)
