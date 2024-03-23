# coding:utf-8

import os
import json
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def load_p_file(f_path):
    with open(f_path, 'r', encoding="utf-8") as f_in:
        idx = 0
        val_list = []
        for line in f_in.readlines():
            idx += 1
            if 23 <= idx <= 120022:
                # 读入的是字符串，所以用 eval 函数获取到字符串运算的返回值
                val_list.append(float(eval(line)))

    # （120000，）
    return np.asarray(val_list).astype(np.float32)


def load_U_file(f_path):
    with open(f_path, 'r', encoding="utf-8") as f_in:
        idx = 0
        val_list = []
        for line in f_in.readlines():
            idx += 1
            if 23 <= idx <= 120022:
                # print("line: {};".format(line) )
                vals = list(map(lambda x: float(eval(x)),
                                line.strip("\n").strip("(").strip(")").split(" ")
                                )
                            )
                val_list.append(vals)
    # （120000，3）
    u_array = np.asarray(val_list).astype(np.float32)

    return u_array


def main():
    def arr_minmax(in_u):
        return (in_u - in_u.min()) / (in_u.max() - in_u.min())

    in_folder = "./Sun_case/"
    step_p_array_l = []
    step_U_array_l = []

    # 1～9
    for step in range(1, 10, 1):
        p_fname = "{}/p".format(step)
        p_file_path = os.path.join(in_folder, p_fname)
        U_fname = "{}/U".format(step)
        U_file_path = os.path.join(in_folder, U_fname)
        p_array = load_p_file(p_file_path)
        U_array = load_U_file(U_file_path)

        # 扩展完每个 size 都是（1，120000）
        step_p_array_l.append(np.expand_dims(p_array, axis=0))
        # 扩展完每个 size 都是（1，120000，3）
        step_U_array_l.append(np.expand_dims(U_array, axis=0))

    step_p = np.concatenate(step_p_array_l, axis=0)  # [9, 120000]
    step_U = np.concatenate(step_U_array_l, axis=0)  # [9, 120000, 3]
    re_shape = [40, 30, 100]
    step_p = np.reshape(step_p, [9] + re_shape)
    step_U = np.reshape(step_U, [9] + re_shape + [3])

    # step_p = arr_minmax(step_p)
    # step_U = arr_minmax(step_U)

    step_list = [x / 10.0 for x in range(1, 10, 1)]
    x_list = [x / 40.0 for x in range(0, 40, 1)]
    y_list = [x / 30.0 for x in range(0, 30, 1)]
    z_list = [x / 100.0 for x in range(0, 100, 1)]

    # itertools.product 产生后面可迭代对象的笛卡尔积，返回的每一个对象都是一个 tuple，每个 tuple
    # 含有四个元素
    # 9*40*30*100
    step_xyz_list = list(itertools.product(*[step_list, x_list, y_list, z_list]))

    # [9 * 40 * 30 * 100, 4]
    step_xyz = np.asarray(step_xyz_list).reshape([9, 40, 30, 100, 4])

    p_mean, p_std = np.mean(step_p, axis=(0, 1, 2, 3)), np.std(step_p, axis=(0, 1, 2, 3))
    U_mean, U_std = np.mean(step_U, axis=(0, 1, 2, 3)), np.std(step_U, axis=(0, 1, 2, 3))
    xyz_mean, xyz_std = np.mean(step_xyz, axis=(0, 1, 2, 3)), np.std(step_xyz, axis=(0, 1, 2, 3))

    step_p = (step_p - p_mean) / p_std
    # print(step_p, step_p.mean(), step_p.std())
    step_U = (step_U - U_mean) / U_std
    # print(step_U, step_U.mean(), step_U.std())
    step_xyz = (step_xyz - xyz_mean) / xyz_std
    # print(step_xyz, step_xyz.mean(), step_xyz.std())

    save_folder = "./inputs/"
    os.makedirs(save_folder, exist_ok=True)
    step_p_save_file_path = os.path.join(save_folder, "step_p_3.npy")  # [9, 40, 30, 100]
    np.save(step_p_save_file_path, step_p)
    step_U_save_file_path = os.path.join(save_folder, "step_U_3.npy")  # [9, 40, 30, 100, 3]
    np.save(step_U_save_file_path, step_U)
    step_xyz_save_file_path = os.path.join(save_folder, "step_xyz_3.npy")  # [9, 40, 30, 100, 4]
    np.save(step_xyz_save_file_path, step_xyz)

    print("step_p:", step_p.shape)
    print("step_U:", step_U.shape)
    print("step_xyz:", step_xyz.shape)


def plot_main(choose: str = 'p'):
    # Define dimensions
    re_shape = [40, 30, 100]  # OK; maybe right shape

    Ny, Nx, Nz = re_shape
    print("Nx:{}, Ny:{}, Nz:{} ".format(Nx, Ny, Nz))
    # 生成三维网格
    X, Y, Z = np.meshgrid(np.arange(Ny), np.arange(Nx), -np.arange(Nz))
    print("X.shape: {} ".format(X.shape))
    print("Y.shape: {} ".format(Y.shape))
    print("Z.shape: {} ".format(Z.shape))

    if choose == 'p':
        # 压力
        in_folder = "./Sun_case/"
        p_fname = "{}/p".format(2)
        p_file_path = os.path.join(in_folder, p_fname)
        p_array = load_p_file(p_file_path)
        print("p_array.shape: {} ".format(p_array.shape))
        data = np.reshape(p_array, re_shape)  # [120000, ] --> [40, 30, 100]
        data = np.transpose(data, [1, 0, 2])  # [40, 30, 100] --> [30, 40, 100]
        # 在最后一个维度进行翻转
        data = np.flip(data, axis=-1)
        print("data.shape: {} ".format(data.shape))
    elif choose == 'U':
        # 速度
        in_folder = "./Sun_case/"
        U_fname = "{}/U".format(2)
        U_file_path = os.path.join(in_folder, U_fname)
        U_array = load_U_file(U_file_path)
        print("U_array.shape: {} ".format(U_array.shape))
        # 在一后一个维度进行平方后相加，猜测是是x,y,z三个方向上的速度分量，最后公式为 √（x^2+y^2+z^2）
        U_array = np.sqrt(np.sum(U_array ** 2, axis=-1))
        data = np.reshape(U_array, re_shape)  # [120000, ] --> [40, 30, 100]
        data = np.transpose(data, [1, 0, 2])  # [40, 30, 100] --> [30, 40, 100]
        data = np.flip(data, axis=-1)
        print("data.shape: {} ".format(data.shape))

    kw = {
        'vmin': data.min(),
        'vmax': data.max(),
        'levels': np.linspace(data.min(), data.max(), 20),
    }

    # Create a figure with 3D ax
    fig = plt.figure(figsize=(8, 6))
    # 创建三维图像
    ax = fig.add_subplot(111, projection='3d')

    # 画三维等高线
    # zdir 用于投影，将其在 zdir 维度上压缩投影到其他两个维度上
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

    # 保存图片
    if choose == 'p':
        line = p_fname.split('/')
        p_or_U = line[0] + line[1] + '.png'
    elif choose == 'U':
        line = U_fname.split('/')
        p_or_U = line[0] + line[1] + '.png'

    plt_dir = './pic/'
    os.makedirs(plt_dir, exist_ok=True)
    plt_save_path = os.path.join(plt_dir, p_or_U)
    plt.savefig(plt_save_path)

    # plt.show()


# 函数入口
if __name__ == "__main__":
    main()
    # plot_main('U')
