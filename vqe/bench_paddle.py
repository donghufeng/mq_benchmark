import os
os.environ["OMP_NUMBER_THREADS"] = "3"

from typing import Tuple
import paddle
import paddle_quantum as pq
from paddle_quantum import qchem as pq_qchem
import warnings
warnings.filterwarnings("ignore")

import time

def geo_trans(data) -> Tuple[str, str]:
    avail_molecule = {
        "HH" : "H2",
        "LiH": "LiH",
        "BeHH": "BeH2",
    }
    xs = []
    s_xs = ""
    for atom in data:
        name = atom[0]
        pos = atom[1]
        s = f"{name} {pos[0]} {pos[1]} {pos[2]}"
        xs.append(s)
        s_xs += name
    if s_xs not in avail_molecule:
        raise ValueError(f"Not support for {data}")
    else:
        return ("; ".join(xs), avail_molecule[s_xs])

def get_qubits_electrons(molecule_name: str) -> Tuple[int, int]:
    molecule_structure = {
        "H2": (4, 2),
        "LiH": (12, 4),
        "BeH2": (14, 6),
    }
    if molecule_structure.get(molecule_name, None) is None:
        raise ValueError(f"Please affend infomation of {molecule_name}.")
    else:
        return molecule_structure[molecule_name]

def bench(data, iter_num):
    # 定义氢分子的几何结构，长度单位为埃
    # h2_geometry = "H 0.0 0.0 0.0; H 0.0 0.0 0.74"
    geometry, molecule_name = geo_trans(data)
    basis_set = "sto-3g"
    multiplicity = 1
    charge = 0

    # 构建 UCCSD 线路.
    # n_qubits = 4
    # n_electrons = 2
    n_qubits, n_electrons = get_qubits_electrons(molecule_name)
    uccsd_ansatz = pq_qchem.UCCSDModel(n_qubits, n_electrons, n_trotter_steps=3)

    # 设置损失函数
    loss_fn = pq_qchem.MolEnergyLoss(geometry, basis_set)

    # 选择 paddlepaddle 中的 Adam 优化器
    optimizer = paddle.optimizer.Adam(
        parameters=uccsd_ansatz.parameters(), 
        learning_rate=0.1
    )

    # 制备初始量子态, e.g. |0000>
    init_state = pq.state.computational_basis(n_qubits, 0)

    print("Start training...")
    start_time = time.time()
    for itr in range(0, iter_num):
        # 运行量子线路得到末态
        state = uccsd_ansatz(init_state)
        # 计算损失函数，即期望值
        loss = loss_fn(state)
        # 反向传播梯度
        loss.backward()
        # 通过loss值更新参数
        optimizer.minimize(loss)
        # 清除当前梯度
        optimizer.clear_grad()
        if itr % 5 == 0:
            print(f"The iter is {itr:3d}, loss is {loss.item():3.5f}.")
    # print("The theoretical value is -1.137283834485513.")
    end_time = time.time()
    print(f"Used time: {end_time - start_time}")
    return end_time - start_time

if __name__ == "__main__":
    data = [["H", [0, 0, 0]], ["H", [0, 0, 0.74]]]
    iter_num = 100
    bench(data, iter_num)
