# QAOA

## Usage

- 根据需要修改 `qaoa_benchmark.py`，终端运行命令 `python qaoa_benchmark.py N_NODES N_LAYERS SHOTS ITER_NUM`。其中`N_NODES`为量子比特数，`N_LAYERS`为QAOA线路层数，`SHOTS`为样本数，ITER_NUM为训练迭代次数。
- 各个框架的代码位于 `benchmark_NAME.py` 下，其中 `NAME` 为框架名，例如：`mindquantum`、`paddle`。
- 所有依赖包及其版本位于`requirements.txt`文件中，运行终端命令`pip install -r requirements.txt`可安装所有依赖包。

## Result

Time for 100 iterations, second.
||MindQuantum|TensorFlow Quantum|PaddlePaddle Quantum|Qiskit|QPanda|Pennylane|
|-|-|-|-|-|-|-|-|
|4 qubits(6 layers)|-|-|-|-|-|-|-|
|8 qubits(6 layers)|-|-|-|-|-|-|-|
|12 qubits(6 layers)|-|-|-|-|-|-|-|
|16 qubits(6 layers)|-|-|-|-|-|-|-|