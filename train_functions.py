import copy
import torch
import os


def generate_metrics_list(metrics_def):
    # 创建一个空字典，用于存储各指标的值列表
    result_dict = {}

    # 遍历输入的 metrics_def 字典中的指标名称
    for name in metrics_def.keys():
        # 将每个指标名称作为键，对应的值初始化为空列表
        result_dict[name] = []

    # 返回生成的字典
    return result_dict


def epoch(scope, loader, on_batch=None, training=False):
    # 从 scope 中获取必要的参数和配置
    model = scope["model"]
    optimizer = scope["optimizer"]
    loss_func = scope["loss_func"]
    metrics_def = scope["metrics_def"]

    # 复制一份 scope，避免修改原始 scope 对象
    scope = copy.copy(scope)
    # 将 loader 添加到 scope 中
    scope["loader"] = loader

    # 生成用于记录各指标值的字典
    metrics_list = generate_metrics_list(metrics_def)
    total_loss = 0

    # 将模型切换到训练或验证模式
    if training:
        model.train()
    else:
        model.eval()

    # 遍历数据加载器中的每个 batch
    for tensors in loader:
        # 如果定义了 process_batch 函数，对数据进行处理
        if "process_batch" in scope and scope["process_batch"] is not None:
            tensors = scope["process_batch"](tensors)

        # 如果定义了 device，将数据移动到指定设备（GPU 或 CPU）
        if "device" in scope and scope["device"] is not None:
            tensors = [tensor.to(scope["device"]) for tensor in tensors]

        # 计算模型输出和损失
        loss, output = loss_func(model, tensors)

        # 如果是训练阶段，进行反向传播和优化器更新
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 累计总损失值
        total_loss += loss.item()

        # 将当前 batch 的信息添加到 scope 中
        scope["batch"] = tensors
        scope["loss"] = loss
        scope["output"] = output
        scope["batch_metrics"] = {}

        # 遍历每个指标并计算 on_batch 函数的值
        for name, metric in metrics_def.items():
            value = metric["on_batch"](scope)
            scope["batch_metrics"][name] = value
            metrics_list[name].append(value)

        # 如果定义了 on_batch 函数，调用该函数
        if on_batch is not None:
            on_batch(scope)

    # 将 metrics_list 添加到 scope 中
    scope["metrics_list"] = metrics_list

    # 计算 on_epoch 函数的值，得到最终的 metrics 字典
    metrics = {}
    for name in metrics_def.keys():
        scope["list"] = scope["metrics_list"][name]
        metrics[name] = metrics_def[name]["on_epoch"](scope)

    # 返回总损失值和最终的 metrics 字典
    return total_loss, metrics


def save_model(scope, epoch):
    try:
        # 检查保存目录是否存在，如果不存在则创建
        save_directory = './saved_models/'
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # 构建文件路径
        save_file_path = os.path.join(save_directory, f"saved_model_epoch{epoch}.pt")

        # 保存模型状态
        torch.save(scope['model'].state_dict(), save_file_path)
        print(f"模型成功保存：{save_file_path}")
    except Exception as e:
        print(f"保存模型时发生错误：{e}")



def train(scope, train_dataset, val_dataset, patience=10, batch_size=256, print_function=print, eval_model=None,
          on_train_batch=None, on_val_batch=None, on_train_epoch=None, on_val_epoch=None, after_epoch=None, save_model=None):
    epochs = scope["epochs"]
    model = scope["model"]
    metrics_def = scope["metrics_def"]
    scope = copy.copy(scope)

    scope["best_train_metric"] = None
    scope["best_train_loss"] = float("inf")
    scope["best_val_metrics"] = None
    scope["best_val_loss"] = float("inf")
    scope["best_model"] = None

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    skips = 0



    for epoch_id in range(1, epochs + 1):
        scope["epoch"] = epoch_id
        print_function("Epoch #" + str(epoch_id))
        # Training
        scope["dataset"] = train_dataset
        train_loss, train_metrics = epoch(scope, train_loader, on_train_batch, training=True)
        scope["train_loss"] = train_loss
        scope["train_metrics"] = train_metrics
        print_function("\tTrain Loss = " + str(train_loss))

        for name in metrics_def.keys():
            print_function("\tTrain " + metrics_def[name]["name"] + " = " + str(train_metrics[name]))
        if on_train_epoch is not None:
            on_train_epoch(scope)
        del scope["dataset"]
        # Validation
        scope["dataset"] = val_dataset
        with torch.no_grad():
            val_loss, val_metrics = epoch(scope, val_loader, on_val_batch, training=False)
        scope["val_loss"] = val_loss
        scope["val_metrics"] = val_metrics
        print_function("\tValidation Loss = " + str(val_loss))
        for name in metrics_def.keys():
            print_function("\tValidation " + metrics_def[name]["name"] + " = " + str(val_metrics[name]))
        if on_val_epoch is not None:
            on_val_epoch(scope)
        del scope["dataset"]
        # Selection
        is_best = None
        if eval_model is not None:
            is_best = eval_model(scope)
        if is_best is None:
            is_best = val_loss < scope["best_val_loss"]
        if is_best:
            scope["best_train_metric"] = train_metrics
            scope["best_train_loss"] = train_loss
            scope["best_val_metrics"] = val_metrics
            scope["best_val_loss"] = val_loss
            scope["best_model"] = copy.deepcopy(model)



            print_function("Model saved!")
            skips = 0
        else:
            skips += 1


        if epoch_id % 30 == 0:
                # 保存模型状态
                if save_model is not None:
                    save_model(scope, epoch_id)

    return scope["best_model"], scope["best_train_metric"], scope["best_train_loss"],\
           scope["best_val_metrics"], scope["best_val_loss"]



def train_model(model, loss_func, train_dataset, val_dataset, optimizer, process_batch=None, eval_model=None,
                on_train_batch=None, on_val_batch=None, on_train_epoch=None, on_val_epoch=None, after_epoch=None,
                epochs=100, batch_size=256, patience=10, device=0, **kwargs):
    model = model.to(device)
    scope = {}
    scope["model"] = model
    scope["loss_func"] = loss_func
    scope["train_dataset"] = train_dataset
    scope["val_dataset"] = val_dataset
    scope["optimizer"] = optimizer
    scope["process_batch"] = process_batch
    scope["epochs"] = epochs
    scope["batch_size"] = batch_size
    scope["device"] = device
    metrics_def = {}
    names = []

    for key in kwargs.keys():
        parts = key.split("_")
        if len(parts) == 3 and parts[0] == "m":
            if parts[1] not in names:
                names.append(parts[1])
    for name in names:
        if "m_" + name + "_name" in kwargs and "m_" + name + "_on_batch" in kwargs and "m_" + name + "_on_epoch" in kwargs:
            metrics_def[name] = {
                "name": kwargs["m_" + name + "_name"],
                "on_batch": kwargs["m_" + name + "_on_batch"],
                "on_epoch": kwargs["m_" + name + "_on_epoch"],
            }
        else:
            print("Warning: " + name + " metric is incomplete!")
    scope["metrics_def"] = metrics_def

    def save_model(scope, epoch):
        try:
            # 检查保存目录是否存在，如果不存在则创建
            save_directory = './saved_models/'
            if not os.path.exists(save_directory):
                os.makedirs(save_directory)

            # 构建文件路径
            save_file_path = os.path.join(save_directory, f"saved_model_epoch{epoch}.pt")

            # 保存模型状态
            torch.save(scope['model'].state_dict(), save_file_path)
            print(f"模型成功保存：{save_file_path}")
        except Exception as e:
            print(f"保存模型时发生错误：{e}")

    return train(scope, train_dataset, val_dataset, eval_model=eval_model, on_train_batch=on_train_batch,
                 on_val_batch=on_val_batch, on_train_epoch=on_train_epoch, on_val_epoch=on_val_epoch,
                 after_epoch=after_epoch, batch_size=batch_size, patience=patience, save_model=save_model)
