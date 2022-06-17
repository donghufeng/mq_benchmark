import numpy as np
from pyqpanda import *


def qaoa_layer(qubit_list, Hamiltonian, beta, gamma):
    vqc = VariationalQuantumCircuit()
    for i in range(len(Hamiltonian)):
        tmp_vec = []
        item = Hamiltonian[i]
        dict_p = item[0]
        for iter in dict_p:
            if 'Z' != dict_p[iter]:
                pass
            tmp_vec.append(qubit_list[iter])

        coef = item[1]

        if 2 != len(tmp_vec):
            pass

        vqc.insert(VariationalQuantumGate_CNOT(tmp_vec[0], tmp_vec[1]))
        vqc.insert(VariationalQuantumGate_RZ(tmp_vec[1], 2 * gamma * coef))
        vqc.insert(VariationalQuantumGate_CNOT(tmp_vec[0], tmp_vec[1]))

    for j in qubit_list:
        vqc.insert(VariationalQuantumGate_RX(j, 2.0 * beta))
    return vqc


def bench(hyperparams={}):
    """
    Performs QAOA optimizations.

    Args:
            hyperparams (dict): hyperparameters to configure this benchmark

                    * 'graph': Graph represented as a NetworkX Graph class

                    * 'n_layers': Number of layers in the QAOA circuit

                    * 'shots': The number of samples.
    """

    graph = hyperparams['graph']
    n_layers = hyperparams['n_layers']
    shots = hyperparams['shots']

    edges = list(graph.edges)
    n_qubits = len(graph.nodes)

    problem_ham = {}
    for (u, v) in edges:
        k = 'Z' + str(u) + ' Z' + str(v)
        if k in problem_ham:
            problem_ham[k] += 1.0
        else:
            problem_ham[k] = 1.0

    ham = PauliOperator(problem_ham)

    beta = var(np.ones((n_layers, 1), dtype='float64'), True)
    gamma = var(np.ones((n_layers, 1), dtype='float64'), True)

    machine = init_quantum_machine(QMachineType.CPU)
    qubit_list = machine.qAlloc_many(n_qubits)
    vqc = VariationalQuantumCircuit()

    for i in qubit_list:
        vqc.insert(VariationalQuantumGate_H(i))

    for i in range(n_layers):
        vqc.insert(qaoa_layer(qubit_list, ham.toHamiltonian(1), beta[i], gamma[i]))

    loss = qop(vqc, ham, machine, qubit_list)  # 问题哈密顿量的期望
    optimizer = MomentumOptimizer.minimize(loss, 0.05, 0.9)  # 使用梯度下降优化器来优化参数

    leaves = optimizer.get_variables()

    steps = 100
    for i in range(steps):
        optimizer.run(leaves, 0)
        loss_value = optimizer.get_loss()
        if (i + 1) % 10 == 0:
            print("training step:", i + 1, "  loss:", "%.4f" % loss_value)

    # 验证结果
    prog = QProg()
    qcir = vqc.feed()
    prog.insert(qcir)
    directly_run(prog)

    result = quick_measure(qubit_list, shots)
    print("Measurement result with QPanda:\n", result)
