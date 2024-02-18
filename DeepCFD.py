import os
import json
import pickle
import pandas as pd
from train_functions import *
from functions import *
from torch.utils.data import TensorDataset
from Models.UNetEx import UNetEx
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def Loading_dataset():
    #load_path = r"D:\dataset\data_16\inputs\340"
    load_path = r"/home/epm_315/syl/learn_python/deep_cfd/dataset/340"
    data_x = np.load(os.path.join(load_path, "input_feats.npy"))
    data_y = np.load(os.path.join(load_path, "input_uv_labels.npy"))
    data_x = data_x.astype(np.float32)
    data_y = data_y.astype(np.float32)
    data_y = torch.tensor(data_y)

    test_x = np.load(os.path.join(load_path, "input_feats_test.npy"))
    test_y = np.load(os.path.join(load_path, "input_uv_labels_test.npy"))
    test_x = test_x.astype(np.float32)
    test_y = test_y.astype(np.float32)
    test_y = torch.tensor(test_y)
    # 将测试集数据移到设备上
    test_x = torch.tensor(test_x).to(device)
    test_y = torch.tensor(test_y).to(device)

    return data_x, data_y, test_x, test_y

# def Spliting_dataset():
#
#     x, y = Loading_dataset()
#     print("x.shape", x.shape)
#     print("y.shape", y.shape)
#     # Spliting dataset into 70% train and 30% test
#     train_data, test_data = split_tensors(x, y, ratio=0.8)
#
#     train_data = [torch.tensor(array).detach() for array in train_data]
#     test_data = [torch.tensor(array).detach() for array in test_data]
#
#     train_dataset, test_dataset = TensorDataset(*train_data), TensorDataset(*test_data)
#     test_x, test_y = test_dataset[:]
#     return train_dataset, test_dataset, test_x, test_y



def after_epoch(scope):
    train_loss_curve.append(scope["train_loss"])
    test_loss_curve.append(scope["val_loss"])
    train_mse_curve.append(scope["train_metrics"]["mse"])
    test_mse_curve.append(scope["val_metrics"]["mse"])
    train_ux_curve.append(scope["train_metrics"]["ux"])
    test_ux_curve.append(scope["val_metrics"]["ux"])
    train_uy_curve.append(scope["train_metrics"]["uy"])
    test_uy_curve.append(scope["val_metrics"]["uy"])


def loss_func(model, batch):
    x, y = batch
    x = x.to(device)  # 设备是你模型所在的设备，例如 'cuda:0'
    y = y.to(device)
    model = model.to(device)
    output = model(x)

    lossu = ((output[:, 0, :, :] - y[:, 0, :, :]) ** 2).reshape(
        (output.shape[0], 1, output.shape[2], output.shape[3]))
    lossv = ((output[:, 1, :, :] - y[:, 1, :, :]) ** 2).reshape(
        (output.shape[0], 1, output.shape[2], output.shape[3]))

    # 计算 Ux、Uy 和 p 方向上的损失
    # output[:, i, :, :] 表示模型预测输出的第 i 个通道
    # y[:, i, :, :] 表示实际标签的第 i 个通道
    # 损失使用平方差（MSE）计算，对于 p 方向使用绝对值平方误差

    # 计算损失
    loss = (lossu + lossv) / channels_weights
    return torch.sum(loss), output

def train_data(epochs, batch_size):
    global train_loss_curve, test_loss_curve, train_mse_curve, test_mse_curve, train_ux_curve, test_ux_curve, train_uy_curve, test_uy_curve
    DeepCFD, train_metrics, train_loss, test_metrics, test_loss = train_model(
        model,  # 使用的模型
        loss_func,  # 定义的损失函数
        train_dataset,  # 训练数据集
        test_dataset,  # 测试数据集
        optimizer,  # 优化器
        epochs=epochs,  # 训练的总轮次
        batch_size=batch_size,  # 每批次的样本数量
        device=device,  # 使用的设备（GPU 或 CPU）
        m_mse_name="Total MSE",  # 总的均方误差指标名称
        m_mse_on_batch=lambda scope: float(torch.sum((scope["output"] - scope["batch"][1]) ** 2)),  # 计算每个批次的均方误差
        m_mse_on_epoch=lambda scope: sum(scope["list"]) / len(scope["dataset"]),  # 计算每个 epoch 的均方误差
        m_ux_name="Ux MSE",  # Ux 方向的均方误差指标名称
        m_ux_on_batch=lambda scope: float(
            torch.sum((scope["output"][:, 0, :, :] - scope["batch"][1][:, 0, :, :]) ** 2)),  # 计算每个批次的 Ux 方向的均方误差
        m_ux_on_epoch=lambda scope: sum(scope["list"]) / len(scope["dataset"]),  # 计算每个 epoch 的 Ux 方向的均方误差
        m_uy_name="Uy MSE",  # Uy 方向的均方误差指标名称
        m_uy_on_batch=lambda scope: float(
            torch.sum((scope["output"][:, 1, :, :] - scope["batch"][1][:, 1, :, :]) ** 2)),  # 计算每个批次的 Uy 方向的均方误差
        m_uy_on_epoch=lambda scope: sum(scope["list"]) / len(scope["dataset"]),  # 计算每个 epoch 的 Uy 方向的均方误差
        patience=25,  # 用于 early stopping 的耐心值
        after_epoch=save_model  # 每个 epoch 后执行的回调函数
    )

    # 将损失数据保存到全局变量中
    train_loss_curve.append(train_loss)  # 注意这里使用 append 而不是 extend
    test_loss_curve.append(test_loss)
    train_mse_curve.append(train_metrics["mse"])
    test_mse_curve.append(test_metrics["mse"])
    train_ux_curve.append(train_metrics["ux"])
    test_ux_curve.append(test_metrics["ux"])
    train_uy_curve.append(train_metrics["uy"])
    test_uy_curve.append(test_metrics["uy"])

    metrics = {}
    metrics["train_metrics"] = train_metrics  # 训练集上的指标数据
    metrics["train_loss"] = train_loss  # 训练集上的损失值，是通过损失函数计算得到的
    metrics["test_metrics"] = test_metrics  # 测试集上的指标数据，可能包括均方误差（MSE）、Ux 方向的均方误差、Uy 方向的均方误差、p 方向的均方误差等
    metrics["test_loss"] = test_loss  # 测试集上的损失值
    curves = {}
    curves["train_loss_curve"] = train_loss_curve
    curves["test_loss_curve"] = test_loss_curve
    curves["train_mse_curve"] = train_mse_curve
    curves["test_mse_curve"] = test_mse_curve
    curves["train_ux_curve"] = train_ux_curve
    curves["test_ux_curve"] = test_ux_curve
    curves["train_uy_curve"] = train_uy_curve
    curves["test_uy_curve"] = test_uy_curve

    config["metrics"] = metrics
    config["curves"] = curves
    # 将训练和测试阶段的损失曲线以及 MSE、Ux MSE、Uy MSE、p MSE 等曲线保  存到 curves 字典中。
    # 将 metrics 和 curves 字典整合到一个名为 config 的字典中。

    new_folder = './result'  # 替换为您的新文件夹路径
    # 确保新文件夹存在，如果不存在则创建
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)

    # 创建训练集和测试集的数据框
    train_metrics_df = pd.DataFrame({'Train Metrics': train_metrics})

    train_metrics_df = pd.DataFrame(train_metrics, index=[0])

    # test_metrics_df = pd.DataFrame(test_metrics)
    test_metrics_df = pd.DataFrame(test_metrics, index=[0])

    # 创建训练集和测试集的曲线数据框
    train_curves_df = pd.DataFrame(curves)
    test_curves_df = pd.DataFrame(curves)
    with open(simulation_directory + "results.json", "w") as file:
        json.dump(config, file)



def test_model(model, test_x, test_y, device, epoch_id):
    model.eval()
    with torch.no_grad():
        out = model(test_x.to(device))
        error = torch.abs(out.cpu() - test_y.cpu())

    s = 0
    print("test_y.shape", test_y.shape)
    print("out.shape", out.shape)

    # 可视化预测结果
    visualize(test_y.cpu().detach().numpy(), out.cpu().detach().numpy(), error.cpu().detach().numpy(), s, epoch_id)



if __name__ == "__main__":

    x, y, _, _ = Loading_dataset()
    _, _, test_x, test_y = Loading_dataset()

    # 创建 PyTorch 的 TensorDataset
    train_dataset = TensorDataset(torch.tensor(x), torch.tensor(y))
    test_dataset = TensorDataset(torch.tensor(test_x), torch.tensor(test_y))

    # 打印数据形状
    print("Training set size:", len(train_dataset))
    print("Testing set size:", len(test_dataset))

    print("y.shape", y.shape)
    channels_weights = torch.sqrt(torch.mean(y.permute(0, 2, 3, 1).reshape((804 * 250 * 50, 2)) ** 2, dim=0)). \
        view(1, -1, 1, 1).to(device)



    simulation_directory = "./Run/"
    if not os.path.exists(simulation_directory):
        os.makedirs(simulation_directory)
    torch.manual_seed(0)
    lr = 0.001
    kernel_size = 5
    filters = [8, 16, 32, 32]
    bn = False
    wn = False
    wd = 0.003
    model = UNetEx(2, 2, filters=filters, kernel_size=kernel_size, batch_norm=bn, weight_norm=wn)
    # Define opotimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    config = {}
    train_loss_curve = []
    test_loss_curve = []
    train_mse_curve = []
    test_mse_curve = []
    train_ux_curve = []
    test_ux_curve = []
    train_uy_curve = []
    test_uy_curve = []

    #Training model
    #训练模型
    epochs = 100100  # 训练的总轮次
    batch_size = 100

    #train_data(epochs, batch_size)
    # # 保存损失函数
    # loss_data = {
    #     'Epoch': list(range(1, len(train_loss_curve) + 1)),
    #     'Train Loss': train_loss_curve,
    #     'Train Total MSE': train_mse_curve,
    #     'Train Ux MSE': train_ux_curve,
    #     'Train Uy MSE': train_uy_curve,
    #     'Test Loss': test_loss_curve,
    #     'Test Total MSE': test_mse_curve,
    #     'Test Ux MSE': test_ux_curve,
    #     'Test Uy MSE': test_uy_curve,
    # }
    #
    # loss_df = pd.DataFrame(loss_data)
    #
    # # 保存 DataFrame 到 Excel 文件
    # excel_filename = "loss_values.xlsx"
    # excel_path = os.path.join(".", excel_filename)
    # loss_df.to_excel(excel_path, index=False)
    #
    # print("Loss values saved to:", excel_path)

    # 加载模型并进行测试
    for epoch_id in range(epochs):
        if epoch_id % 300 == 0 and epoch_id > 00:
            loaded_model = UNetEx(2, 2, filters=filters, kernel_size=kernel_size, batch_norm=bn, weight_norm=wn)
            loaded_model.load_state_dict(
                torch.load(f'./saved_models/saved_model_epoch{epoch_id}.pt', map_location=device))
            loaded_model = loaded_model.to(device)
            test_x = test_x.to(device)

            test_model(loaded_model, test_x, test_y, device, epoch_id)
