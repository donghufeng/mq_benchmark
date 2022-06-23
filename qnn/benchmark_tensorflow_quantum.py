import time
import tensorflow as tf
import tensorflow_quantum as tfq
from mindquantum.core import Circuit as mcircuit
from mindquantum import Hamiltonian, QubitOperator

import cirq
import numpy as np
from data_generator import circle_data_point_generator
from tran_mcircuit import mq_to_tfq

def bench(n_qubits, epoch, batch, train_samples):

    Ntrain = train_samples
    Ntest = 100
    gap = 0.01

    # 生成数据集
    train_features, train_labels, test_features, test_labels = circle_data_point_generator(
        Ntrain, Ntest, gap, n_qubits, 1
    )

    qreg = cirq.LineQubit.range(n_qubits)

    # 编码输入特征
    def convert_to_circuit(features, n_qubits, qreg, pr_table):
        mencoder = mcircuit()
        for i, value in enumerate(features):
            mencoder.ry(value, i)
        return mq_to_tfq.trans_circuit_mindquantum_cirq(mencoder, n_qubits, qreg, pr_table)


    pr_table = dict()
    x_train_circ = [convert_to_circuit(x, n_qubits, qreg, pr_table) for x in train_features]
    x_test_circ = [convert_to_circuit(x, n_qubits, qreg, pr_table) for x in test_features]

    x_train_tfcirc = tfq.convert_to_tensor(x_train_circ)
    x_test_tfcirc = tfq.convert_to_tensor(x_test_circ)

    #构建ansatz
    mansatz = mcircuit()
    for i in range(n_qubits):
        mansatz.rz(f"beta_1_{i}", i)
        mansatz.ry(f"beta_2_{i}", i)
        mansatz.rz(f"beta_3_{i}", i)

    for i in range(n_qubits - 1):
        mansatz.x(i + 1, i)
    mansatz.x(0, n_qubits - 1)

    for i in range(n_qubits):
        mansatz.ry(f"beta_4_{i}", i)

    pr_table = dict()
    circ = mq_to_tfq.trans_circuit_mindquantum_cirq(mansatz, n_qubits, qreg, pr_table)

    # 生成哈密顿量
    hamiltonian_QubitOp = Hamiltonian(QubitOperator('Z0'))
    ham = mq_to_tfq.trans_hamiltonian(hamiltonian_QubitOp, qreg)

    # 生成模型
    model = tf.keras.Sequential([
        # The input is the data-circuit, encoded as a tf.string
        tf.keras.layers.Input(shape=(), dtype=tf.string),
        # The PQC layer returns the expected value of the hamiltonian, range [-1,1].
        tfq.layers.PQC(circ, operators=ham),
    ])

    model.compile(
        loss=tf.keras.losses.MeanAbsoluteError(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.1))

    print(model.summary())

    # 训练模型
    print("tensorflow-quantum: start training")
    start_time = time.time()
    qnn_history = model.fit(
        x_train_tfcirc, train_labels,
        batch_size=batch,
        epochs=epoch, verbose=1)
    end_time = time.time()
    print("tensorflow-quantum: finish")

    '''
    # 测试模型准确率
    qnn_results = model.evaluate(x_test_tfcirc, test_labels)

    acc = 0
    res = model(x_test_tfcirc)
    print(res)
    for i in range(len(res)):
        if res[i] > 0:
            predict = 1
        else:
            predict = -1
        if predict == test_labels[i]:
            acc += 1

    print(acc / len(test_labels))
    '''

    return end_time - start_time

if __name__=='__main__':
    res = bench(4, 5, 20, 200)
    print(res)