import paddle
from paddle_quantum.ansatz import Circuit
from paddle_quantum.loss import ExpecVal
from paddle_quantum import Hamiltonian
import warnings
warnings.filterwarnings("ignore")


def circuit_QAOA(n_qubits, n_layers, edges, nodes):
    # 初始化 n 个量子比特的量子电路
    circ = Circuit(n_qubits)
    # 制备量子态 |s>
    circ.superposition_layer()
    # 搭建 n_layers 层 ansatz 线路
    circ.qaoa_layer(edges, nodes, n_layers)

    return circ


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
    nodes = list(graph.nodes)
    n_qubits = len(graph.nodes)

    circ = circuit_QAOA(n_qubits, n_layers, edges, nodes)

    SEED = 1024  # 设置全局随机数种子
    paddle.seed(SEED)

    LR = 0.05  # 梯度下降的学习率
    opt = paddle.optimizer.Adam(learning_rate=LR, parameters=circ.parameters())  # 使用 Adam 优化器

    # 以 list 的形式构建哈密顿量
    ham_list = []
    for (u, v) in edges:
        ham_list.append([1.0, 'z' + str(u) + ',z' + str(v)])
    # print(ham_list)

    # 构造损失函数
    loss_func = ExpecVal(Hamiltonian(ham_list))

    # 训练
    steps = hyperparams['iter_num']
    for i in range(steps):
        state = circ()
        # 计算梯度并优化
        loss = loss_func(state)
        loss.backward()
        opt.minimize(loss)
        opt.clear_grad()
        if (i + 1) % 10 == 0:
            print("training step:", i + 1, "  loss:", "%.4f" % loss.numpy())

    # 测量
    state = circ()
    probs_measured = state.measure(shots=shots, plot=False)
    print("Probability distribution of measurement result with PaddleQuantum:\n", probs_measured)
