import pyqpanda as pq
import numpy as np
import time
from data_generator import circle_data_point_generator
from tran_mcircuit import mq_to_qpanda
from mindquantum.core import Circuit as mcircuit
from mindquantum import Hamiltonian, QubitOperator


def bench(n_qubits, epoch, batch, train_samples):

    Ntrain = train_samples
    Ntest = 100
    gap = 0.01

    # 生成数据集
    (
        train_features,
        train_labels,
        test_features,
        test_labels,
    ) = circle_data_point_generator(Ntrain, Ntest, gap, n_qubits, 1)

    machine = pq.init_quantum_machine(pq.QMachineType.CPU)
    qubit_list = machine.qAlloc_many(n_qubits)

    # 生成encoder
    mencoder = mcircuit()
    for i in range(n_qubits):
        mencoder.rz(f"alpha_{i}", i)

    # 生成ansatz
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

    # 生成哈密顿量
    hamiltonian_QubitOp = Hamiltonian(QubitOperator("Z0"))
    ham = mq_to_qpanda.trans_hamiltonian(hamiltonian_QubitOp)

    # 生成qpanda线路
    qnn, pr_dict = mq_to_qpanda.trans_circuit_mindquantum_qpanda(
        mencoder + mansatz, n_qubits, machine, qubit_list
    )

    label = pq.var(np.float64(train_labels[0]))
    for j in range(len(mencoder.params_name)):
        pr_dict[mencoder.params_name[j]].set_value(train_features[0, j])

    loss = pq.poly(pq.qop(qnn, ham, machine, qubit_list) - label, pq.var(2))
    optimizer = pq.AdamOptimizer.minimize(loss, 0.1, 0.9, 0.999, 1.0e-6)
    leaves = set()
    for pr in mansatz.params_name:
        leaves.add(pr_dict[pr])

    # 训练参数
    start_time = time.time()
    print("Qpanda: start training")
    for i in range(epoch):
        for k in range(train_samples // batch):
            for l in range(batch):
                for j in range(len(mencoder.params_name)):
                    pr_dict[mencoder.params_name[j]].set_value(
                        train_features[k * batch + l, j]
                    )
                label.set_value(train_labels[k * batch + l])
                optimizer.run(leaves, 0)

        loss_value = optimizer.get_loss()
        print(loss_value)
    end_time = time.time()
    print("Qpanda: finish")
    return end_time - start_time

if __name__=='__main__':
    res = bench(4, 5, 20, 200)
    print(res)