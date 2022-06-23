from cProfile import label
import pennylane as qml
from pennylane import numpy as np
from data_generator import circle_data_point_generator
import time


def bench(n_qubits, epoch, batch, train_samples):

    Ntrain = train_samples
    Ntest = 100
    gap = 0.01
    n_wires = n_qubits

    # 生成数据集
    (
        train_features,
        train_labels,
        test_features,
        test_labels,
    ) = circle_data_point_generator(Ntrain, Ntest, gap, n_wires, 1)
    train_features = np.array(train_features, requires_grad=False)
    train_labels = np.array(train_labels, requires_grad=False)
    test_features = np.array(test_features, requires_grad=False)
    test_labels = np.array(test_labels, requires_grad=False)

    dev = qml.device("default.qubit", wires=n_wires, shots=None)

    # 生成ansatz
    def my_ansatz(params, n_wires):
        for i in range(n_wires):
            qml.RZ(params[0 + 4 * i], wires=i)
            qml.RY(params[1 + 4 * i], wires=i)
            qml.RZ(params[2 + 4 * i], wires=i)
        for i in range(n_wires - 1):
            qml.CNOT(wires=[i, i + 1])
        qml.CNOT(wires=[n_wires - 1, 0])
        for i in range(n_wires):
            qml.RY(params[3 + 4 * i], wires=i)

    # 生成总线路
    @qml.batch_input(argnum=[0])
    @qml.qnode(dev)
    def circuit(input, ansatz_pr, n_wires):
        qml.AngleEmbedding(input, range(n_wires), rotation="Y")
        my_ansatz(ansatz_pr, n_wires)
        return qml.expval(qml.PauliZ(0))

    batch_size = batch
    epochs = epoch
    samples = train_samples

    ansatz_pr = []
    for j in range(4 * n_wires):
        ansatz_pr.append(1)

    ansatz_pr = np.array(ansatz_pr, requires_grad=True)

    opti = qml.AdamOptimizer()

    def loss_fn(input, ansatz_pr, labels):
        result = circuit(input, ansatz_pr, n_wires)
        loss = (result - labels) ** 2
        return np.mean(loss)

    print("pennylane: start training")
    start_time = time.time()
    # 训练网络
    for i in range(epochs):
        for j in range(samples // batch_size):
            encoder_pr = train_features[batch_size * j : batch_size * j + batch_size]
            labels = train_labels[batch_size * j : batch_size * j + batch_size]
            res = opti.step(loss_fn, encoder_pr, ansatz_pr, labels)
            ansatz_pr = res[-2]
    end_time = time.time()
    print("pennylane: finish")

    """
    # 测试准确率
    acc=0
    for i in range(len(test_features)):
        if circuit([test_features[i]], ansatz_pr, n_wires)<0:
            predict = -1
        else:
            predict = 1
        if predict == test_labels[i]:
            acc+=1
    print(acc/len(test_features))
    """

    return end_time - start_time

if __name__=='__main__':
    res = bench(4, 5, 20, 200)
    print(res)