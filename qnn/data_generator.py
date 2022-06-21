import numpy as np

# 圆形决策边界两分类数据集生成器
def circle_data_point_generator(Ntrain, Ntest, boundary_gap, n_qubits, seed_data):
    """
    :param Ntrain: 训练集大小
    :param Ntest: 测试集大小
    :param boundary_gap: 取值于 (0, 0.5), 两类别之间的差距
    :param seed_data: 随机种子
    :return: 四个列表：训练集x，训练集y，测试集x，测试集y
    """
    # 生成共Ntrain + Ntest组数据，x对应n维数据点，y对应编号
    # 取前Ntrain个为训练集，后Ntest个为测试集
    train_x, train_y = [], []
    num_samples, seed_para = 0, 0
    avg_lenth = np.sqrt(0.25 * n_qubits)  # 计算生成向量的平均长度
    while num_samples < Ntrain + Ntest:
        np.random.seed((seed_data + 10) * 1000 + seed_para + num_samples)
        data_point = np.random.rand(n_qubits) * 2 - 1  # 生成[-1, 1]范围内n维向量

        # 如果数据点的模小于(avg_lenth - gap)，标为-1
        if np.linalg.norm(data_point) < avg_lenth - boundary_gap / 2:
            train_x.append(data_point)
            train_y.append(-1)
            num_samples += 1

        # 如果数据点的模大于(avg_lenth + gap)，标为1
        elif np.linalg.norm(data_point) > avg_lenth + boundary_gap / 2:
            train_x.append(data_point)
            train_y.append(1.0)
            num_samples += 1

        else:
            seed_para += 1

    train_x = np.array(train_x).astype("float32")
    train_y = np.array(train_y).astype("int").T

    print(
        "训练集的维度大小 x {} 和 y {}".format(
            np.shape(train_x[0:Ntrain]), np.shape(train_y[0:Ntrain])
        )
    )
    print(
        "测试集的维度大小 x {} 和 y {}".format(
            np.shape(train_x[Ntrain:]), np.shape(train_y[Ntrain:])
        ),
        "\n",
    )

    return train_x[0:Ntrain], train_y[0:Ntrain], train_x[Ntrain:], train_y[Ntrain:]
