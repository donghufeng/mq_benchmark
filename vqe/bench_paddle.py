import os
os.environ["OMP_NUMBER_THREADS"] = "3"

import paddle
import paddle_quantum.qchem as qchem
from paddle_quantum.loss import ExpecVal
from paddle_quantum import Hamiltonian
from paddle_quantum.state import zero_state, State
from paddle_quantum.ansatz import Circuit
from paddle_quantum.linalg import dagger
from paddle_quantum import Backend

import matplotlib.pyplot as plt

import numpy
from numpy import pi as PI
from numpy import savez, zeros

# 无视警告
import warnings
warnings.filterwarnings("ignore")

import time

def U_theta(num_qubits: int, depth: int) -> Circuit:
    """
    Quantum Neural Network
    """
    
    # 按照量子比特数量/网络宽度初始化量子神经网络
    cir = Circuit(num_qubits)
    
    # 内置的 {R_y + CNOT} 电路模板
    cir.real_entangled_layer(depth = depth)
    
    # 铺上最后一列 R_y 旋转门
    cir.ry()
        
    return cir

class StateNet(paddle.nn.Layer):
    """
    Construct the model net
    """

    def __init__(self, num_qubits: int, depth: int, init_state, loss_func):
        super(StateNet, self).__init__()
        
        self.depth = depth
        self.num_qubits = num_qubits
        self.cir = U_theta(self.num_qubits, self.depth)

        self.init_state = init_state
        self.loss_func = loss_func
        
    # 定义损失函数和前向传播机制
    def forward(self):
        
        # 运行电路
        state = self.cir(self.init_state)
        # 计算损失函数
        loss = self.loss_func(state)     

        return loss, self.cir

def bench(data, iter_num):
    geo = qchem.geometry(structure=data)
    # geo = qchem.geometry(file='h2.xyz')

    # 将分子信息存储在 molecule 里，包括单体积分（one-body integrations），双体积分（two-body integrations），分子的哈密顿量等
    molecule = qchem.get_molecular_data(
        geometry=geo,
        basis='sto-3g',
        charge=0,
        multiplicity=1,
        method="fci",
        # if_save=True,
        # if_print=True
    )
    # 提取哈密顿量
    molecular_hamiltonian = qchem.spin_hamiltonian(molecule=molecule,
                                                filename=None, 
                                                multiplicity=1, 
                                                mapping_method='jordan_wigner',)
    # 打印结果
    # print("\nThe generated h2 Hamiltonian is \n", molecular_hamiltonian)

    ITR = iter_num  # 设置训练的总迭代次数
    LR = 0.4   # 设置学习速率
    D = 2      # 设置量子神经网络中重复计算模块的深度 Depth
    N = molecular_hamiltonian.n_qubits # 设置参与计算的量子比特数

    # 定义初始态
    init_state = zero_state(N)

    # 定义损失函数
    loss_func = ExpecVal(molecular_hamiltonian)

    # 确定网络的参数维度
    net = StateNet(N, D, init_state, loss_func)

    # 一般来说，我们利用Adam优化器来获得相对好的收敛，
    # 当然你可以改成SGD或者是RMS prop.
    opt = paddle.optimizer.Adam(learning_rate=LR, parameters=net.parameters())

    # 记录优化结果
    summary_iter, summary_loss = [], []

    print("Start training.")
    start_time = time.time()

    # # 优化循环
    for itr in range(1, ITR + 1):

        # 前向传播计算损失函数
        loss, cir = net()

        # 在动态图机制下，反向传播极小化损失函数
        loss.backward()
        opt.minimize(loss)
        opt.clear_grad()

        # 更新优化结果
        summary_loss.append(loss.numpy())
        summary_iter.append(itr)

        # 打印结果
        if itr % 5 == 0:
            print("iter:", itr, "loss:", "%.4f" % loss.numpy())
            print("iter:", itr, "Ground state energy:", "%.4f Ha" 
                                                % loss.numpy())
        if itr == ITR:
            end_time = time.time()
            print(f"Used time: {end_time-start_time}")
            return end_time - start_time
            # print("\n训练后的电路：") 
            # print(cir)

    # 储存训练结果到 output 文件夹
    # os.makedirs("output", exist_ok=True)
    # savez("./output/summary_data", iter = summary_iter, 
    #                             energy=summary_loss)



if __name__ == "__main__":
    # data = [['H', [-0., 0., 0.0]], ['H', [-0., 0., 0.74]]]
    data = [["Li", [0, 0, 0]], ["H", [1, 0, 0]]]
    bench(data, 50)