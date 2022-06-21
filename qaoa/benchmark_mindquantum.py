from mindquantum.core import Circuit, Hamiltonian
from mindquantum.simulator import Simulator
from mindquantum.algorithm.nisq.qaoa import MaxCutAnsatz
from mindquantum.framework import MQAnsatzOnlyLayer
import mindspore.nn as nn
import mindspore as ms


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
    ham = Hamiltonian(-maxcut.hamiltonian)
    # print(circ)
    # print(ham)

    # 搭建待训练量子神经网络以获取最优参数
    ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")

    sim = Simulator('projectq', circ.n_qubits)

    grad_ops = sim.get_expectation_with_grad(ham, circ)  # 获取计算变分量子线路的期望值和梯度的算子

    net = MQAnsatzOnlyLayer(grad_ops)  # 生成待训练的神经网络
    opti = nn.Adam(net.trainable_params(), learning_rate=0.05)  # 设置针对网络中所有可训练参数、学习率为0.05的Adam优化器
    train_net = nn.TrainOneStepCell(net, opti)  # 对神经网络进行一步训练

    steps = hyperparams['iter_num']
    for i in range(steps):
        cut = (len(graph.edges) - train_net()) / 2  # 将神经网络训练一步并计算得到的结果（切割边数）。注意：每当'train_net()'运行一次，神经网络就训练了一步
        if i % 10 == 0:
            print("train step:", i, ", cut:", cut)  # 每训练10步，打印当前训练步数和当前得到的切割边数

    params_dict = dict(zip(circ.params_name, net.weight.asnumpy()))  # 获取训练得到的最优参数
    # print(params_dict)

    # 测量
    circ.measure_all()  # 为线路中所有比特添加测量门
    result = sim.sampling(circ, pr=params_dict, shots=shots)
    print("Measurement result with MindQuantum:\n", result)
