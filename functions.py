import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from typing import Tuple, Dict


# 切分数据集，目前没用
def split_tensors(*tensors: Tuple, ratio: float) -> Tuple:
    """
    @param tensors: x, y
    @param ratio: 切分比例
    @return: 切分后的 train, test 数据集
    """

    assert len(tensors) > 0
    split1, split2 = [], []
    count = len(tensors[0])
    for tensor in tensors:
        assert len(tensor) == count
        split1.append(tensor[:int(len(tensor) * ratio)])
        split2.append(tensor[int(len(tensor) * ratio):])
    if len(tensors) == 1:
        split1, split2 = split1[0], split2[0]
    return split1, split2


# 网络 Xavier 初始化，防止梯度消失，目前没用
def initialize(model: nn.Module, gain: int = 1, std: float = 0.02) -> None:
    """
    @param model: 模型
    @param gain: xavier 初始化参数
    @param std: 标准差
    """

    for module in model.modules():
        if type(module) in [nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d]:
            nn.init.xavier_normal_(module.weight, gain)
            if module.bias is not None:
                nn.init.normal_(module.bias, 0, std)


# 数据可视化函数
def visualize(sample_y: np.array = None, out_y: np.array = None, error: np.array = None, epoch_id: int = None,
              all_path: Dict = None, test_interval: int = None) -> None:
    """
    @param test_interval: 测试间隔
    @param sample_y: 测试数据的 label
    @param out_y: 模型输出的预测结果
    @param error: sample_y 和 out_y 的绝对差值
    @param epoch_id: 当前 epoch 轮次
    @param all_path: 全部的存储路径
    """
    minu = -0.01
    maxu = 0.01
    minv = -0.01
    maxv = 0.01
    mineu = -0.01
    maxeu = 0.01
    minev = -0.01
    maxev = 0.01

    plot_path = all_path["plot"]
    dat_path = all_path["dat"]

    def avoid_duplicate(plt=None, data_in: np.array = None, channel: int = None, title: str = None,
                        y_label: str = None, min_in: float = None, max_in: float = None) -> None:
        """
        @param max_in: maxu or maxv or maxev or maxeu
        @param min_in: minu or minv or minev or mineu
        @param data_in: 区分是 sample，out_y 还是 error
        @param channel: 区分是 x 通道还是 y 通道
        @param y_label: 纵坐标名称
        @param title: 子图名称
        @param plt: 子图画轴
        """

        plt.imshow(np.transpose(data_in[0, channel, :, :]), cmap='jet', vmin=min_in, vmax=max_in, origin='lower',
                   extent=[0, 250, 0, 50])
        cbar = plt.colorbar(orientation='horizontal', shrink=0.9)
        # 获取 Colorbar 的刻度标签对象
        labels = cbar.ax.get_xticklabels()
        # 设置刻度标签的旋转角度
        for label in labels:
            label.set_rotation(-45)

        if y_label is not None:
            plt.ylabel(y_label, fontsize=18)
        if title is not None:
            plt.title(title, fontsize=22)

    # 画 contrast 图
    plt.figure()
    fig = plt.gcf()
    fig.set_size_inches(15, 6)

    plt.subplot(2, 3, 1)
    avoid_duplicate(plt, sample_y, 0, "CFD", "Ux", minu, maxu)

    plt.subplot(2, 3, 2)
    avoid_duplicate(plt, out_y, 0, "CNN", None, minu, maxu)

    plt.subplot(2, 3, 3)
    avoid_duplicate(plt, error, 0, "ERROR", None, mineu, maxeu)

    plt.subplot(2, 3, 4)
    avoid_duplicate(plt, sample_y, 1, None, "Uy", minv, maxv)

    plt.subplot(2, 3, 5)
    avoid_duplicate(plt, out_y, 1, None, None, minv, maxv)

    plt.subplot(2, 3, 6)
    avoid_duplicate(plt, error, 1, None, None, minev, maxev)

    plt.tight_layout()
    plt.savefig(plot_path + '/' + f"contrast{epoch_id}.png")
    plt.close()

    for i in range(sample_y.shape[0]):
        if i == 200:
            # 画预测与真实的流速对比图
            # Get data for visualization
            # 这里目前我不知道为什么一定是 200，如果指定取最后一个，没必要 for 循环直接指定就行了，奇怪的逻辑
            true_val = sample_y[i, :, :, :]
            pred_val = out_y[i, :, :, :]

            # 获取数值序列，类似 range，不过这里不是 int，都是 float
            # 这里因为 x 的维度为 250，y 的维度为 50，所以现在这么设置
            x = np.linspace(0, 250, 250)
            y = np.linspace(0, 50, 50)

            # 获取网格采样矩阵，x 的维度为 m，y 的维度为 n 的情况下，生成的 X，Y 都是 （n，m）矩阵
            # 在这里就是（50，250），其中 X 的每行都是 x，Y 的每列都是 y
            X, Y = np.meshgrid(x, y)
            # 获取 label中 ux 和 uy 两个维度，命名为 u，v
            u = true_val[0, :, :]
            v = true_val[1, :, :]
            u_pred = pred_val[0, :, :]
            v_pred = pred_val[1, :, :]

            # Plot and save images
            plt.figure(figsize=(10, 7))

            plt.subplot(2, 1, 1)
            plt.title("True", fontsize=20)
            plt.streamplot(X, Y, u, v, density=2, linewidth=1.5, arrowstyle='->', color='black')
            plt.xlabel('X', fontsize=16)
            plt.ylabel('Y', fontsize=16)
            plt.xlim(0, X.max())
            plt.ylim(0, Y.max())

            plt.subplot(2, 1, 2)
            plt.title("Prediction", fontsize=20)
            plt.streamplot(X, Y, u_pred, v_pred, density=2, linewidth=1.5, arrowstyle='->', color='black')
            plt.xlabel('X', fontsize=16)
            plt.ylabel('Y', fontsize=16)
            plt.xlim(0, X.max())
            plt.ylim(0, Y.max())

            plt.tight_layout()
            plt.savefig(plot_path + '/' + f"speed_contrast_epoch{epoch_id}_dimension{i}.png")
            plt.close()

            # 画流速对比图
            # 计算真实值和预测值的速度模
            speed = np.sqrt(u ** 2 + v ** 2)  # (50, 250)
            speed_pred = np.sqrt(u_pred ** 2 + v_pred ** 2)  # (50, 250)

            # 计算真实值和预测值之间的速度模误差
            error_speed = abs(speed - speed_pred)  # (50, 250)

            eps = 1e-5
            speed_pred_for_no_zero = speed_pred + eps
            error_relative = error_speed / speed_pred_for_no_zero

            np.savetxt(plot_path + '/' + f"absolute_error_epoch{epoch_id}_dimension{i}.csv", error_speed)
            np.savetxt(plot_path + '/' + f"relative_error_epoch{epoch_id}_dimension{i}.csv", error_relative)

            mine = np.min(error_speed)
            maxe = np.max(error_speed)

            # 找到误差最大值的位置
            # 将一维坐标映射到多维数组
            max_error_index = np.unravel_index(np.argmax(error_speed), error_speed.shape)
            # 这里对坐标进行了变换，先取的第二个维度，再取的第一个维度
            max_error_coordinates = (max_error_index[1], max_error_index[0])
            # 输出最大误差值和对应的位置
            print("最大速度模误差：", maxe)
            print("最大误差位置坐标：", max_error_coordinates)

            # 可视化速度模误差
            plt.figure(figsize=(8, 5))
            plt.xlim(0, 250)
            plt.ylim(0, 50)
            plt.imshow(np.transpose(error_speed), cmap='jet', vmin=mine, vmax=maxe, origin='lower',
                       extent=[0, 250, 0, 50])
            plt.colorbar(orientation='vertical', fraction=0.01, pad=0.01)
            plt.xlabel('X', fontsize=16)
            plt.ylabel('Y', fontsize=16)
            plt.title("Error_speed", fontsize=20)

            plt.tight_layout()
            plt.savefig(plot_path + '/' + f"error_epoch{epoch_id}_dimension{i}.png")
            plt.close()

            # 存储 dat 文件
            error_speed = error_speed.reshape(250, 50)

            if epoch_id == test_interval:
                # Save test data
                HEADER = "TITLE = \"plot\"\nVARIABLES = \"X\",\"Y\",\"U\",\"V\",\"error\"\nZONE I=250, J=50, F=POINT"
                # 切分为 ux 和 uy 两个数组
                channel_test_v = np.split(true_val, 2, axis=0)
                # 分别将 ux 和 uy 两个数组拉平，每个的 shape 都是（50*250, ）
                channel_test_v_flat = [channel.flatten() for channel in channel_test_v]
                # 按列堆叠数组的函数, 将一系列一维数组作为列堆叠成一个二维数组
                # 每一列的 shape 都是（50*250，），一共有 5 列
                data = np.column_stack((X.flatten(), Y.flatten(), *channel_test_v_flat, error_speed.flatten()))
                np.savetxt(dat_path + '/' + f"true_dimension{i}.dat", data, delimiter="\t",
                           header=HEADER, comments='')

            # Save pred data
            HEADER = "TITLE = \"plot\"\nVARIABLES = \"X\",\"Y\",\"U\",\"V\",\"error\"\nZONE I=250, J=50, F=POINT"
            channel_pred_v = np.split(pred_val, 2, axis=0)
            channel_pred_v_flat = [channel.flatten() for channel in channel_pred_v]
            data_pred = np.column_stack((X.flatten(), Y.flatten(), *channel_pred_v_flat, error_speed.flatten()))
            np.savetxt(dat_path + '/' + f"prediction_epoch{epoch_id}_dimension{i}.dat", data_pred, delimiter="\t",
                       header=HEADER, comments='')


# 每个 epoch 之后更新 curve 图，方便实时查看当前结果
def plot_curves(curves: Dict = None, progress_path: str = None) -> None:
    """
    @param progress_path: progress path to save picture
    @param curves: all curves
    """

    # 当前已经训练的 epoch 数目
    cur_epoch = [i for i in range(1, len(curves["train_loss_curve"]) + 1)]
    # 大图对象 fig 以及子图对象 ax
    fig, ax = plt.subplots(3, 4, figsize=(45, 20))

    # 精简代码，避免重复代码累赘
    def avoid_duplicate(axis=None, curve_name: str = None) -> None:
        """
        @param curve_name: 子图纵坐标以及子图名字
        @param axis: 子图画轴
        """

        parts = curve_name.strip('').split('_')
        y_label = parts[0] + '_' + parts[1]

        axis.set_xlabel("epoch", fontsize=15)
        axis.set_ylabel(y_label, fontsize=15)
        axis.set_title(curve_name, fontsize=20)
        # axis.grid(linestyle='-.', linewidth=0.6)

    i, j, k = 0, 0, 0
    cnt = 0
    r_name, r_val = None, None
    for curve_name, curve_value in curves.items():
        if cnt == 12:
            r_name, r_val = curve_name, curve_value
            break

        if i < 4:
            ax[0][i].plot(cur_epoch, curve_value)
            avoid_duplicate(ax[0][i], curve_name)
            i += 1
        elif j < 4:
            ax[1][j].plot(cur_epoch, curve_value)
            avoid_duplicate(ax[1][j], curve_name)
            j += 1
        else:
            ax[2][k].plot(cur_epoch, curve_value)
            avoid_duplicate(ax[2][k], curve_name)
            k += 1

        cnt += 1

    plt.tight_layout()
    plot_curve_path = progress_path + '/' + 'all_curves.png'
    plt.savefig(plot_curve_path)
    print('All curves has been ploted in {}'.format(plot_curve_path))
    plt.close()

    plt.figure()
    plt.plot(cur_epoch, r_val)
    plt.xlabel('epoch')
    plt.ylabel('R²')
    plt.title('R_square')
    plt.tight_layout()
    r_curve_path = progress_path + '/' + 'r_square.png'
    plt.savefig(r_curve_path)
    plt.close()
