import copy
import torch
import os
from torch import nn
from typing import List, Tuple, Dict, Callable, Any
from sklearn.model_selection import KFold
from torch.utils.data import TensorDataset


# 用于计算 R² 指标
def calculate_r_square(pred: torch.Tensor = None, label: torch.Tensor = None) -> float:
    """
    @param pred: 预测值
    @param label: 真实值
    """

    predictions = pred.cpu().detach()
    targets = label.cpu().detach()

    dim_1 = predictions.shape[0]

    # 将数据维度重新排列为（batch_size, -1）形状，以便计算每个样本的R平方
    predictions_flat = predictions.view(dim_1, -1)
    targets_flat = targets.view(dim_1, -1)

    # 计算每个样本的平均值
    mean_targets = torch.mean(targets_flat, dim=1, keepdim=True)

    # 计算总平方和（Total Sum of Squares, TSS）
    TSS = torch.sum((targets_flat - mean_targets) ** 2, dim=1)

    # 计算残差平方和（Residual Sum of Squares, RSS）
    RSS = torch.sum((targets_flat - predictions_flat) ** 2, dim=1)

    # 计算每个样本的R平方
    R_squared = 1 - (RSS / TSS)

    # 对R平方进行求均值，得到整体的R平方值
    overall_R_squared = torch.mean(R_squared)

    return overall_R_squared.item()


# 创建 metric list
def generate_metrics_list(metrics_def: Dict) -> Dict:
    """
    @param metrics_def: metric传入字典
    @return: 输出一个包含各种 key 但是对应 value 为空的字典
    """

    # 创建一个空字典，用于存储各指标的值列表
    result_dict = {}

    # 遍历输入的 metrics_def 字典中的指标名称
    for name in metrics_def.keys():
        # 将每个指标名称作为键，对应的值初始化为空列表
        result_dict[name] = []

    # 返回生成的字典
    return result_dict


def save_model(scope: Dict, epoch: int, save_path: str = None) -> None:
    """
    @param save_path: 存储路径
    @param scope: 参数字典
    @param epoch: 当前训练轮次数
    """

    # 目前的存储只能跑一个程序，跑多了命名是一样的，会覆盖掉之前的模型参数文件
    try:
        # 检查保存目录是否存在，如果不存在则创建
        os.makedirs(save_path, exist_ok=True)

        # 构建文件路径
        save_file_path = os.path.join(save_path, f"saved_model_epoch{epoch}.pt")

        # 保存模型状态
        torch.save(scope['model'].state_dict(), save_file_path)
        print(f"模型成功保存：{save_file_path}")
    except Exception as e:
        print(f"保存模型时发生错误：{e}")


def epoch(scope: Dict, loader, on_batch: Any = None, training: bool = False) -> Tuple:
    """
    @param scope: 参数字典
    @param loader: dataloader
    @param on_batch: 目前没定义
    @param training: 用于判断是训练还是测试
    @return: 总损失值和最终的 metrics 字典
    """

    # 从 scope 中获取必要的参数和配置
    model = scope["model"]
    optimizer = scope["optimizer"]
    loss_func = scope["loss_func"]
    metrics_def = scope["metrics_def"]

    # 将 loader 添加到 scope 中
    scope["loader"] = loader

    # 生成用于记录各指标值的字典
    metrics_list = generate_metrics_list(metrics_def)
    total_loss = 0
    r = 0

    # 将模型切换到训练或验证模式
    if training:
        model.train()
    else:
        model.eval()

    # 遍历数据加载器中的每个 batch，这里分 batch 的时候，如果最后剩下不满足一个 batch_size 大小的数据的话
    # 会直接用一个小 batch，而非凑够 batch_size
    cnt = 0
    for batch in loader:
        # 如果定义了 process_batch 函数，对数据进行处理, 目前没定义
        if "process_batch" in scope and scope["process_batch"] is not None:
            batch = scope["process_batch"](batch)

        # 如果定义了 device，将数据移动到指定设备（GPU 或 CPU）
        if "device" in scope and scope["device"] is not None:
            batch = [tensor.to(scope["device"]) for tensor in batch]

        # 计算模型输出和损失，注意这里的 loss 是一整个 batch 上累加的 loss
        loss, output = loss_func(model, batch)

        r_batch = calculate_r_square(output, batch[1])

        # 如果是训练阶段，进行反向传播和优化器更新
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 累计总损失值
        total_loss += loss.item()
        r += r_batch
        cnt += 1

        # 将当前 batch 的信息添加到 scope 中
        scope["batch"] = batch
        # 注意这里存入的不是 total_loss，而是一个 batch 上的 loss
        # scope["loss"] = loss  # 这个值会一直被覆盖掉，直到最后一个 batch 的 batch_loss 看起来也没啥用
        scope["output"] = output

        # 遍历每个指标并计算 on_batch 函数的值
        for name, metric in metrics_def.items():
            # 这里的 metric 是一个字典，metric["on_batch"]是一个 lambda 匿名函数
            value = metric["on_batch"](scope)
            # scope["batch_metrics"][name] = value
            # 这里是一个 list 对象，会对一个epoch 中的每一个 batch 计算出的 value 值进行存储
            metrics_list[name].append(value)

        # 如果定义了 on_batch 函数，调用该函数
        if on_batch is not None:
            on_batch(scope)

    # 将 metrics_list 添加到 scope 中
    scope["metrics_list"] = metrics_list

    # 计算 on_epoch 函数的值，得到最终的 metrics 字典
    metrics = {}
    for name in metrics_def.keys():
        # 获取到相应 name 的 list，里面存储这一指标在每个 batch 上的值
        scope["list"] = scope["metrics_list"][name]
        metrics[name] = metrics_def[name]["on_epoch"](scope)

    # loss = total_loss / cnt
    r /= cnt

    # print('metric:', metrics)
    # 返回总损失值和最终的 metrics 字典
    return total_loss, metrics, r


def train(scope: Dict, train_dataset, test_dataset, patience: int = 10, batch_size: int = 256, print_function=print,
          eval_model=None, on_train_batch=None, on_val_batch=None, on_train_epoch=None, on_val_epoch=None,
          after_epoch=None, save_model: Any = None, all_path: Dict = None, test_model: Any = None,
          test_interval: int = None, warm_up: Any = None, lr_decay: Any = None) -> Tuple:
    """
    @param lr_decay: 学习率衰减函数
    @param warm_up: 学习率预热函数
    @param test_interval: 测试间隔
    @param test_model: 测试模型函数
    @param all_path: 全部用到的存储路径
    @param scope: 包含各种训练参数的字典
    @param train_dataset: 训练集
    @param test_dataset: 测试集
    @param patience: 忍耐值，目前没使用
    @param batch_size: 批数据大小
    @param print_function: 输出函数，实际上就是 print...
    @param eval_model: 测试模型
    @param on_train_batch: 目前没定义
    @param on_val_batch: 目前没定义
    @param on_train_epoch: 目前没定义
    @param on_val_epoch: 目前没定义
    @param after_epoch: 回调函数，用于记录训练结果
    @param save_model: 保存模型函数
    @return: 五个最优参数
    """

    # 从字典中取出相应参数，方便使用
    epochs = scope["epochs"]
    normal_epochs = scope["normal_epochs"]
    warm_up_iter = scope["warm_up_iter"]
    model = scope["model"]
    metrics_def = scope["metrics_def"]
    num_folds = scope["kf"]
    mode = scope["mode"]

    # 新定义 5 个 key 值用于记录最优参数
    scope["best_train_metric"] = None
    scope["best_train_loss"] = float("inf")
    scope["best_test_metrics"] = None
    scope["best_test_loss"] = float("inf")
    scope["best_model"] = None
    scope["r"] = None

    k_fold = KFold(n_splits=num_folds, shuffle=True)
    per_fold_epochs = epochs // num_folds
    fold_cre_epoch = 0
    for fold, (train_indices, val_indices) in enumerate(k_fold.split(train_dataset[0])):

        train_dataset_in = TensorDataset(train_dataset[0][train_indices], train_dataset[1][train_indices])
        val_dataset_in = TensorDataset(train_dataset[0][val_indices], train_dataset[1][val_indices])

        # 创建数据加载器
        train_loader = torch.utils.data.DataLoader(train_dataset_in, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset_in, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # 从 1 开始，方便理解，否则默认是从 0 开始的
        best_epoch = 0
        for epoch_id in range(1, per_fold_epochs + 1):
            current_epoch = epoch_id + fold_cre_epoch
            # 记录当前训练的轮次数
            scope["epoch"] = current_epoch
            print_function("Epoch #" + str(current_epoch))

            # Training
            scope["dataset"] = train_dataset_in  # 用于计算 mse_on_epoch 指标
            # 运行当前 epoch 训练，返回的 train_loss 是一个 total_loss
            train_loss, train_metrics, _ = epoch(scope, train_loader, on_train_batch, training=True)
            # 记录当前 epoch 的损失以及其他指标
            scope["train_loss"] = train_loss
            scope["train_metrics"] = train_metrics
            print_function("\tTrain Loss = " + str(train_loss))
            for name in metrics_def.keys():
                print_function("\tTrain " + metrics_def[name]["name"] + " = " + str(train_metrics[name]))
            if on_train_epoch is not None:
                on_train_epoch(scope)
            del scope["dataset"]

            # Validation
            scope["dataset"] = val_dataset_in
            # 不计算梯度，不进行模型更新
            with torch.no_grad():
                val_loss, val_metrics, _ = epoch(scope, val_loader, on_val_batch, training=False)
            scope["val_loss"] = val_loss
            scope["val_metrics"] = val_metrics
            print_function("\tVal Loss = " + str(val_loss))
            for name in metrics_def.keys():
                print_function("\tVal " + metrics_def[name]["name"] + " = " + str(val_metrics[name]))
            if on_val_epoch is not None:
                on_val_epoch(scope)
            del scope["dataset"]

            # Test
            scope["dataset"] = test_dataset
            # 不计算梯度，不进行模型更新
            with torch.no_grad():
                test_loss, test_metrics, r = epoch(scope, test_loader, None, training=False)
            scope["test_loss"] = test_loss
            scope["test_metrics"] = test_metrics
            scope["r"] = r
            print_function("\tTest Loss = " + str(test_loss))
            for name in metrics_def.keys():
                print_function("\tTest " + metrics_def[name]["name"] + " = " + str(test_metrics[name]))
            if on_val_epoch is not None:
                on_val_epoch(scope)
            del scope["dataset"]

            # Selection the best metric
            # 目前也是一次训练之后就选择一次最优指标
            is_best = None
            if eval_model is not None:
                is_best = eval_model(scope)
            if is_best is None:
                is_best = test_loss < scope["best_test_loss"]
            # 只要当前模型的 test loss 更小，则更新最优指标
            if is_best:
                scope["best_train_metric"] = train_metrics
                scope["best_train_loss"] = train_loss
                scope["best_test_metrics"] = test_metrics
                scope["best_test_loss"] = test_loss
                scope["best_model"] = copy.deepcopy(model)
                best_epoch = current_epoch

                print_function("Best model has been selected !")

            if mode == 'sy':
                if current_epoch <= warm_up_iter:
                    warm_up(current_epoch)
                elif current_epoch > normal_epochs:
                    lr_decay(current_epoch)

            # 这里是每个 epoch 记录一次
            if after_epoch is not None:
                after_epoch(scope=scope, epoch_id=current_epoch, test_mode=True)

            # 存储模型并且进行测试
            if current_epoch % test_interval == 0:
                # 保存模型状态
                if save_model is not None:
                    save_path = all_path['saved_models']
                    save_model(scope, current_epoch, save_path)
                if test_model is not None:
                    test_model(model, current_epoch)

        fold_cre_epoch += per_fold_epochs

    # # 创建 dataloader 自动输出批量数据用于训练
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    #
    # # 从 1 开始，方便理解，否则默认是从 0 开始的
    # best_epoch = 0
    # for epoch_id in range(1, epochs + 1):
    #     # 记录当前训练的轮次数
    #     scope["epoch"] = epoch_id
    #     print_function("Epoch #" + str(epoch_id))
    #
    #     # Training
    #     scope["dataset"] = train_dataset  # 用于计算 mse_on_epoch 指标
    #     # 运行当前 epoch 训练，返回的 train_loss 是一个 total_loss
    #     train_loss, train_metrics = epoch(scope, train_loader, on_train_batch, training=True)
    #     # 记录当前 epoch 的损失以及其他指标
    #     scope["train_loss"] = train_loss
    #     scope["train_metrics"] = train_metrics
    #     print_function("\tTrain Loss = " + str(train_loss))
    #     for name in metrics_def.keys():
    #         print_function("\tTrain " + metrics_def[name]["name"] + " = " + str(train_metrics[name]))
    #     if on_train_epoch is not None:
    #         on_train_epoch(scope)
    #     del scope["dataset"]
    #
    #     # Test
    #     # 现在的逻辑是每次训练之后都验证一次，但是这里的验证集就是测试集 多少还是不太合理
    #     scope["dataset"] = test_dataset
    #     # 不计算梯度，不进行模型更新
    #     with torch.no_grad():
    #         test_loss, test_metrics = epoch(scope, test_loader, on_val_batch, training=False)
    #     scope["test_loss"] = test_loss
    #     scope["test_metrics"] = test_metrics
    #     print_function("\tTest Loss = " + str(test_loss))
    #     for name in metrics_def.keys():
    #         print_function("\tTest " + metrics_def[name]["name"] + " = " + str(test_metrics[name]))
    #     if on_val_epoch is not None:
    #         on_val_epoch(scope)
    #     del scope["dataset"]
    #
    #     # Selection the best metric
    #     # 目前也是一次训练之后就选择一次最优指标
    #     is_best = None
    #     if eval_model is not None:
    #         is_best = eval_model(scope)
    #     if is_best is None:
    #         is_best = test_loss < scope["best_test_loss"]
    #     # 只要当前模型的 test loss 更小，则更新最优指标
    #     if is_best:
    #         scope["best_train_metric"] = train_metrics
    #         scope["best_train_loss"] = train_loss
    #         scope["best_test_metrics"] = test_metrics
    #         scope["best_test_loss"] = test_loss
    #         scope["best_model"] = copy.deepcopy(model)
    #         best_epoch = epoch_id
    #
    #         print_function("Best model has been selected !")
    #
    #     if epoch_id <= warm_up_iter:
    #         warm_up(epoch_id)
    #     elif epoch_id > normal_epochs:
    #         lr_decay(epoch_id)
    #
    #     # 这里是每个 epoch 记录一次
    #     if after_epoch is not None:
    #         after_epoch(scope=scope, epoch_id=epoch_id)
    #
    #     # 存储模型并且进行测试
    #     if epoch_id % test_interval == 0:
    #         # 保存模型状态
    #         if save_model is not None:
    #             save_path = all_path['saved_models']
    #             save_model(scope, epoch_id, save_path)
    #         if test_model is not None:
    #             test_model(model, epoch_id)

    return scope["best_model"], scope["best_train_metric"], scope["best_train_loss"], \
        scope["best_test_metrics"], scope["best_test_loss"], best_epoch


def train_model(model: nn.Module, loss_func, train_dataset, val_dataset, optimizer, process_batch=None,
                eval_model: nn.Module = None, on_train_batch: Any = None, on_val_batch: Any = None,
                on_train_epoch: Any = None, on_val_epoch: Any = None, after_epoch: Any = None,
                epochs: int = 100, normal_epochs: int = 100, warm_up_iter: int = 20, batch_size: int = 256,
                patience: int = 10, device=0, all_path: Dict = None, test_model: Any = None, test_interval: int = None,
                warm_up: Any = None, lr_decay: Any = None, kf: int = None, mode: str = None, **kwargs) -> Any:
    """
    @param mode: 模型选择
    @param kf: k-折交叉验证
    @param lr_decay: 学习率衰减函数
    @param warm_up: 学习率预热函数
    @param warm_up_iter: 学习率预热轮次
    @param normal_epochs: 峰值训练轮次
    @param test_interval: 测试间隔
    @param test_model: 测试模型函数
    @param all_path: 全部用到的存储路径
    @param model: 模型
    @param loss_func: 损失函数
    @param train_dataset: 训练数据集
    @param val_dataset: 验证集（目前就是测试集）
    @param optimizer: 优化器
    @param process_batch: 数据预处理函数，这里目前没有定义
    @param eval_model: 测试模型，这里目前没有定义
    @param on_train_batch: 目前没有定义
    @param on_val_batch: 目前没有定义
    @param on_train_epoch: 目前没有定义
    @param on_val_epoch: 目前没有定义
    @param after_epoch: 数据回调函数，用于记录损失值以及 mse
    @param epochs: 训练总轮数
    @param batch_size: 批量大小
    @param patience: 忍耐值，用于 early_stopping，目前这里没定义
    @param device: 设备
    @param kwargs: 其他未定义的传入参数组合成的字典对象
    @return: 训练函数对象
    """

    model = model.to(device)
    # 定义参数字典 scope
    scope = {"model": model, "loss_func": loss_func, "train_dataset": train_dataset, "val_dataset": val_dataset,
             "optimizer": optimizer, "process_batch": process_batch, "epochs": epochs, "batch_size": batch_size,
             "device": device, "warm_up_iter": warm_up_iter, "normal_epochs": normal_epochs, "kf": kf, "mode": mode}

    # 存储 names 中 存储的 name 相关的参数字典
    metrics_def = {}
    # 存储名字字符串，目前是 mse，ux，uy
    names = []

    # 用于提取出 kwargs 中存在的 string 类型的名字，例如 m_mse_name，m_ux_name...
    for key in kwargs.keys():
        parts = key.split("_")
        if len(parts) == 3 and parts[0] == "m":
            if parts[1] not in names:
                names.append(parts[1])

    # 提取出 kwargs 中的如 m_mse_on_batch， m_mse_on_epoch 等，存入 metrics_def[name]，这个例子的 name 为 mse
    for name in names:
        if "m_" + name + "_name" in kwargs and "m_" + name + "_on_batch" in kwargs and \
                "m_" + name + "_on_epoch" in kwargs:
            metrics_def[name] = {
                "name": kwargs["m_" + name + "_name"],
                "on_batch": kwargs["m_" + name + "_on_batch"],
                "on_epoch": kwargs["m_" + name + "_on_epoch"],
            }
        else:
            print("Warning: " + name + " metric is incomplete!")

    scope["metrics_def"] = metrics_def

    return train(scope, train_dataset, val_dataset, eval_model=eval_model, on_train_batch=on_train_batch,
                 on_val_batch=on_val_batch, on_train_epoch=on_train_epoch, on_val_epoch=on_val_epoch,
                 after_epoch=after_epoch, batch_size=batch_size, patience=patience, save_model=save_model,
                 all_path=all_path, test_model=test_model, test_interval=test_interval, warm_up=warm_up,
                 lr_decay=lr_decay)
