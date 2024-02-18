import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
import os


def split_tensors(*tensors, ratio):
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


def initialize(model, gain=1, std=0.02):
    for module in model.modules():
        if type(module) in [nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d]:
            nn.init.xavier_normal_(module.weight, gain)
            if module.bias is not None:
                nn.init.normal_(module.bias, 0, std)


def visualize(sample_y, out_y, error, s, epoch_id):
    minu = -0.01
    maxu = 0.01
    minv = -0.01
    maxv = 0.01
    mineu = -0.01
    maxeu = 0.01
    minev = -0.01
    maxev = 0.01

    # Create "plot" folder for saving images
    plot_folder = "./plot"
    os.makedirs(plot_folder, exist_ok=True)

    plt.figure()
    fig = plt.gcf()
    fig.set_size_inches(15, 10)
    plt.subplot(2, 3, 1)
    plt.title('CFD', fontsize=18)
    plt.imshow(np.transpose(sample_y[s, 0, :, :]), cmap='jet', vmin=minu, vmax=maxu, origin='lower',
               extent=[0, 250, 0, 50])
    plt.colorbar(orientation='horizontal')
    plt.ylabel('Ux', fontsize=18)

    plt.subplot(2, 3, 2)
    plt.title('CNN', fontsize=18)
    plt.imshow(np.transpose(out_y[s, 0, :, :]), cmap='jet', vmin=minu, vmax=maxu, origin='lower',
               extent=[0, 250, 0, 50])
    plt.colorbar(orientation='horizontal')

    plt.subplot(2, 3, 3)
    plt.title('Error', fontsize=18)
    plt.imshow(np.transpose(error[s, 0, :, :]), cmap='jet', vmin=mineu, vmax=maxeu, origin='lower',
               extent=[0, 250, 0, 50])
    plt.colorbar(orientation='horizontal')

    plt.subplot(2, 3, 4)
    plt.imshow(np.transpose(sample_y[s, 1, :, :]), cmap='jet', vmin=minv, vmax=maxv, origin='lower',
               extent=[0, 250, 0, 50])
    plt.colorbar(orientation='horizontal')
    plt.ylabel('Uy', fontsize=18)
    plt.subplot(2, 3, 5)
    plt.imshow(np.transpose(out_y[s, 1, :, :]), cmap='jet', vmin=minv, vmax=maxv, origin='lower',
               extent=[0, 250, 0, 50])
    plt.colorbar(orientation='horizontal')

    plt.subplot(2, 3, 6)
    plt.imshow(np.transpose(error[s, 1, :, :]), cmap='jet', vmin=minev, vmax=maxev, origin='lower',
               extent=[0, 250, 0, 50])
    plt.colorbar(orientation='horizontal')

    plot_filename = f"contrast{epoch_id}.png"
    plot_path = os.path.join(plot_folder, plot_filename)
    plt.savefig(plot_path)
    plt.close()
    #
    # for i in range(sample_y.shape[0]):
    #     if i % 150 == 0 and i >>0 :
    #         # Get data for visualization
    #         test_val = sample_y[i, :, :, :]
    #         pred_val = out_y[i, :, :, :]
    #         x = np.linspace(0, 250, 250)
    #         y = np.linspace(0, 50, 50)
    #         X, Y = np.meshgrid(x, y)
    #         u_1 = test_val[0, :, :].reshape(250, 50)
    #         v_1 = test_val[1, :, :].reshape(250, 50)
    #
    #         u_pred_1 = pred_val[0, :, :].reshape(250, 50)
    #         v_pred_1 = pred_val[1, :, :].reshape(250, 50)
    #
    #
    #         # 计算真实值和预测值的速度模
    #         speed_1 = np.sqrt(u_1 ** 2 + v_1 ** 2)
    #         speed_pred_1 = np.sqrt(u_pred_1 ** 2 + v_pred_1 ** 2)
    #
    #         # 计算真实值和预测值之间的速度模误差
    #         error_speed_1 = speed_1 - speed_pred_1
    #         error_speed_1 = abs(error_speed_1)
    #         meane_1 = np.mean(error_speed_1)
    #         mine = np.min(error_speed_1)
    #         maxe = np.max(error_speed_1)
    #         # 定义阈值，即5% maxe的误差值
    #         threshold = 0.05 * meane_1
    #
    #         # 循环遍历每个元素，将小于阈值的误差设为0
    #         for m in range(error_speed_1.shape[0]):
    #             for n in range(error_speed_1.shape[1]):
    #                 if abs(error_speed_1[m, n]) < threshold:
    #                     error_speed_1[m, n] = 0
    #
    #         # 找到误差最大值的位置
    #         max_error_index_1 = np.unravel_index(np.argmax(error_speed_1), error_speed_1.shape)
    #         max_error_coordinates_1 = (max_error_index_1[1], max_error_index_1[0])  # 修正坐标顺序
    #
    #         # 输出最大误差值和对应的位置
    #         print("最大速度模误差：", maxe)
    #         print("最大误差位置坐标：", max_error_coordinates_1)

    for i in range(sample_y.shape[0]):

        if i == 200 :
            # Get data for visualization
            true_val = sample_y[i, :, :, :]
            pred_val = out_y[i, :, :, :]
            x = np.linspace(0, 250, 250)
            y = np.linspace(0, 50, 50)
            X, Y = np.meshgrid(x, y)
            u = true_val[0, :, :]
            v = true_val[1, :, :]
            u_data = u.flatten()
            v_data = v.flatten()
            u_pred = pred_val[0, :, :]
            v_pred = pred_val[1, :, :]
            u_data_pred = u_pred.flatten()
            v_data_pred = v_pred.flatten()

            # Plot and save images
            plt.figure(figsize=(10, 4))

            plt.subplot(2, 1, 1)
            plt.title("true")
            plt.streamplot(X, Y, u_data.reshape(X.shape), v_data.reshape(X.shape), density=2, linewidth=1.5,
                           arrowstyle='->', color='black')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.xlim(0, X.max())
            plt.ylim(0, Y.max())

            plt.subplot(2, 1, 2)
            plt.title("pred")
            plt.streamplot(X, Y, u_data_pred.reshape(X.shape), v_data_pred.reshape(X.shape), density=2, linewidth=1.5,
                           arrowstyle='->', color='black')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.xlim(0, X.max())
            plt.ylim(0, Y.max())

            # Save the plot as an image
            plot_filename = f"plot_epoch{epoch_id}_{i}.png"
            plot_path = os.path.join(plot_folder, plot_filename)
            plt.savefig(plot_path)
            plt.close()

            # 计算真实值和预测值的速度模
            speed = np.sqrt(u ** 2 + v ** 2)

            speed_pred = np.sqrt(u_pred ** 2 + v_pred ** 2)


            # 计算真实值和预测值之间的速度模误差
            error_speed = speed - speed_pred
            error_speed = abs(error_speed)
            meane = np.mean(error_speed)
            mine = np.min(error_speed)
            maxe = np.max(error_speed)
            # 找到误差最大值的位置

            max_error_index = np.unravel_index(np.argmax(error_speed), error_speed.shape)
            max_error_coordinates = (max_error_index[1], max_error_index[0])



            # 输出最大误差值和对应的位置
            print("最大速度模误差：", maxe)
            print("最大误差位置坐标：", max_error_coordinates)

            # 定义阈值，即5% maxe的误差值
            threshold = 0.05 * meane

            # # 循环遍历每个元素，将小于阈值的误差设为0
            # for m in range(error_speed.shape[0]):
            #     for n in range(error_speed.shape[1]):
            #         if abs(error_speed[m, n]) < threshold:
            #             error_speed[m, n] = 0

            # 创建一个云图（scatter plot）来可视化速度模误差
            plt.figure(figsize=(8, 6))
            plt.xlim(0, 250)
            plt.ylim(0, 50)
            plt.imshow(np.transpose(error_speed), cmap='jet', vmin=mine, vmax=maxe, origin='lower', extent=[0, 250, 0, 50])

            plt.colorbar(orientation='horizontal')


            # Save the error as an image
            error_filename = f"error_epoch{epoch_id}_{i}.png"
            error_path = os.path.join(plot_folder, error_filename)
            plt.savefig(error_path)
            plt.close()

    print("plot done")

    # 创建 "dat" 文件夹
    dat_folder = "./dat"
    os.makedirs(dat_folder, exist_ok=True)  # 使用 exist_ok=True 可以避免在文件夹已存在时抛出异常

    for i in range(sample_y.shape[0]):
        if i == 200  :
            # Save pred data

            x = np.linspace(0, 250, 250)
            y = np.linspace(0, 50, 50)
            z = y
            X, Y = np.meshgrid(x, y)

            u = out_y[i, 0, :, :].reshape(-1, 1)
            v = out_y[i, 1, :, :].reshape(-1, 1)
            w = out_y[i, 1, :, :].reshape(-1, 1)  # Update to use the correct channel
            p = out_y[i, 1, :, :].reshape(-1, 1)  # Update to use the correct channel
            k = out_y[i, 1, :, :].reshape(-1, 1)  # Update to use the correct channel
            E = out_y[i, 1, :, :].reshape(-1, 1)  # Update to use the correct channel

            # HEADER = "TITLE = \"plot\"\nVARIABLES = \"X\",\"Y\",\"Z\",\"U\",\"V\",\"W\",\"P\",\"K\",\"E\"\nZONE I=250, J=50, F=POINT"
            # channel_pred_v = np.split(pred_val, 2, axis=0)
            # channel_pred_v_flat = [channel.flatten() for channel in channel_pred_v]
            # data2 = np.column_stack((X.flatten(), Y.flatten(), u.flatten(), u.flatten(), v.flatten(), w.flatten(),
            #                          p.flatten(), k.flatten(), E.flatten(),))
            #
            # pred_data_path = os.path.join(dat_folder, f"complete_data_epoch{epoch_id}_{i}.dat")
            # np.savetxt(pred_data_path, data2, delimiter="\t", header=HEADER, comments='')

        if i == 200:

            # 计算真实值和预测值的速度模
            speed = np.sqrt(u ** 2 + v ** 2)
            speed_pred = np.sqrt(u_pred ** 2 + v_pred ** 2)

            # 将 speed 的形状改变为 (50, 250)
            speed = speed.reshape(250, 50)
            speed_pred = speed_pred.reshape(250, 50)

            # 计算真实值和预测值之间的速度模误差
            error_speed = speed - speed_pred
            error_speed = abs(error_speed)
            meane = np.mean(error_speed)
            mine = np.min(error_speed)
            maxe = np.max(error_speed)

            if epoch_id == 300:
                # Save test data
                test_val = sample_y[i, :, :, :]
                x = np.linspace(0, 250, 250)
                y = np.linspace(0, 50, 50)
                X, Y = np.meshgrid(x, y)
                HEADER = "TITLE = \"plot\"\nVARIABLES = \"X\",\"Y\",\"U\",\"V\",\"error\"\nZONE I=250, J=50, F=POINT"
                channel_test_v = np.split(test_val, 2, axis=0)
                channel_test_v_flat = [channel.flatten() for channel in channel_test_v]
                data = np.column_stack((X.flatten(), Y.flatten(), *channel_test_v_flat, error_speed.flatten()))
                test_data_path = os.path.join(dat_folder, f"true_data_epoch{epoch_id}_{i}.dat")
                np.savetxt(test_data_path, data, delimiter="\t", header=HEADER, comments='')


            # Save pred data
            pred_val = out_y[i, :, :, :]

            HEADER = "TITLE = \"plot\"\nVARIABLES = \"X\",\"Y\",\"U\",\"V\",\"error\"\nZONE I=250, J=50, F=POINT"
            channel_pred_v = np.split(pred_val, 2, axis=0)
            channel_pred_v_flat = [channel.flatten() for channel in channel_pred_v]
            data1 = np.column_stack((X.flatten(), Y.flatten(), *channel_pred_v_flat, error_speed.flatten()))
            pred_data_path = os.path.join(dat_folder, f"pred_data_epoch{epoch_id}_{i}.dat")
            np.savetxt(pred_data_path, data1, delimiter="\t", header=HEADER, comments='')

    print("data done")