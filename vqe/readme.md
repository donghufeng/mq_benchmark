# VQE

## qiskit

Problems:

- 只有一个集成的solver，限制迭代次数？
- 运行时间太长，终止条件不确定。



## pennylane

Docs:

- [A brief overview of VQE](https://pennylane.ai/qml/demos/tutorial_vqe.html)
- [qml.UCCSD](https://pennylane.readthedocs.io/en/stable/code/api/pennylane.UCCSD.html)
- [Adaptive circuits for quantum chemistry](https://pennylane.ai/qml/demos/tutorial_adaptive_circuits.html#romero2017)

Problems:

- 怎么确定 `electrons`？是外层电子还是总电子数目？
- 计算 `cost_fn(params)` 的时间太长，对于数据集 `LiH`。

## paddle paddle

Problems:

- 运行环境冲突，只能在docker中运行。

## Qpanda

Docs:

- [变分量子特征求解算法(VQE)](https://qpanda-tutorial.readthedocs.io/zh/latest/VQE.html)

Problems:

- `pyqpanda` 未提供VQE接口。
- CPP的案例不是迭代的算法，与需求不一致，不知道怎么改写。

## mindquantum

Problems：

- 对于数据集 `CH4` ，执行 67 行 `ham.sparse(total_circuit.n_qubits)` 程序闪退。