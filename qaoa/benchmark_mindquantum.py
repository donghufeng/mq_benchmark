from mindquantum.core import Circuit, Hamiltonian
from mindquantum.simulator import Simulator
from mindquantum.algorithm.nisq.qaoa import MaxCutAnsatz
import random
import numpy as np


def gradient_decs(grad_ops, params, steps):
    exp_val_1, g = grad_ops(params)
    # print(exp_val)
    # print(g)
    alpha = 0.05  # 学习率
    for i in range(steps):
        for j in range(len(params)):
            params[j] = params[j] - alpha * g[0][0][j]
        exp_val_2, g = grad_ops(params)
        if exp_val_1[0][0] - exp_val_2[0][0] < 1e-6:
            return params
        if exp_val_2[0][0] < exp_val_1[0][0]:
            exp_val_1 = exp_val_2

    return params


def bench(hyperparams={}):
    """
    Performs QAOA optimizations.

    Args:
            hyperparams (dict): hyperparameters to configure this benchmark

                    * 'graph': Graph represented as a NetworkX Graph class

                    * 'n_layers': Number of layers in the QAOA circuit

                    * 'params': Numpy array of trainable parameters that is fed into the circuit

                    * 'shots': The number of samples.
    """

    graph = hyperparams['graph']
    n_layers = hyperparams['n_layers']
    shots = hyperparams['shots']

    # 生成对应的的量子线路和哈密顿量
    maxcut = MaxCutAnsatz(list(graph.edges), n_layers)

    circ = maxcut.circuit  # 已对所有量子比特作用H门
    ham = Hamiltonian(-maxcut.hamiltonian)  # 生成哈密顿量
    # print(circ)
    # print(ham)

    sim = Simulator('projectq', circ.n_qubits)  # 创建模拟器，backend使用‘projectq’，能模拟5个比特（'circ'线路中包含的比特数）

    if 'params' in hyperparams:
        params = hyperparams['params']
    else:
        # 搭建待训练量子神经网络以获取最优参数
        grad_ops = sim.get_expectation_with_grad(ham, circ)  # 获取计算变分量子线路的期望值和梯度的算子（相当于cost_function）

        params = np.array([random.random() for i in range(len(circ.params_name))], dtype=complex)
        steps = 100
        gradient_decs(grad_ops, params, steps)

    # 获取线路参数
    params_dict = dict(zip(circ.params_name, params))
    # print(params_dict)

    # 测量
    circ.measure_all()  # 为线路中所有比特添加测量门
    result = sim.sampling(circ, pr=params_dict, shots=shots)
    print("Measurement result with MindQuantum:\n", result)
