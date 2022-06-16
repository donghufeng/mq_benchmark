# VQE

## Usage

- 根据需要修改 `bench.py`，终端运行命令 `python bench.py NAME`，其中`NAME`为分子名，例如：`H2`、`LiH`。
- 分子结构数据位于 `./vqe/data/` 下。
- 各个框架的代码位于 `bench_NAME.py` 下，其中 `NAME` 为框架名，例如：`mindquantum`、`paddle`。
- `*.ipynb` 为一些框架测试的代码，无用。

## Result

Time for 50 iterations, second.
||MindQuantum Next Version|MindQuantum|TensorFlow Quantum|PaddlePaddle Quantum|Qiskit|QPanda|Pennylane|
|-|-|-|-|-|-|-|-|
|H2(4 qubits)|<0.01|0.36|5.54|1.71|5.47|3.48|93.02|
|LiH(12 qubits)|0.05|39.9|874.5|197|1428|-|-|
|BeH2(14 qubits)|0.18|253.64|1939|4034|-|-|-|
