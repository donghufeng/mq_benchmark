# QAOA

## Usage

- 根据需要修改 `qaoa_benchmark.py`，终端运行命令 `python qaoa_benchmark.py N_NODES N_LAYERS SHOTS ITER_NUM`。其中`N_NODES`为量子比特数，`N_LAYERS`为QAOA线路层数，`SHOTS`为样本数，ITER_NUM为训练迭代次数。
- 各个框架的代码位于 `benchmark_NAME.py` 下，其中 `NAME` 为框架名，例如：`mindquantum`、`paddle`。
- 所有依赖包及其版本位于`requirements.txt`文件中，运行终端命令`pip install -r requirements.txt`可安装所有依赖包。
- 使用 Google Colab (https://colab.research.google.com) 可直接在浏览器中运行`qaoa_benchmark_on_google_colab.ipynb`文件，进行qaoa benchmark。Note: 该平台无法测试paddle-quantum。

## Result

Time for 100 iterations, second.

|                     | MindQuantum | QPanda | Pennylane | Qiskit | TensorFlow Quantum | Paddle Quantum (on win11) |
| ------------------- | ----------- | ------ | --------- | ------ | ------------------ | ------------------------- |
| 4 qubits(6 layers)  | 1.56        | 11.28  | 159.79    | 1.28   | 9.5                | 12.73                     |
| 6 qubits(6 layers)  | 4           | 42.48  | 490.38    | 9.83   | 14                 | 27.07                     |
| 8 qubits(6 layers)  | 1.54        | 205.02 | 1556.13   | 17.22  | 23.03              | 70.17                     |
| 10 qubits(6 layers) | 1.85        | 456.24 | 2900.4    | 35.24  | 31.95              | 148.45                    |
| 12 qubits(6 layers) | -           | -      | -         | -      | -                  | -                         |

运行环境：

|        | MindQuantum                    | QPanda                         | Pennylane                      | Qiskit                         | TensorFlow Quantum             | Paddle Quantum                                       |
| ------ | ------------------------------ | ------------------------------ | ------------------------------ | ------------------------------ | ------------------------------ | ---------------------------------------------------- |
| 系统   | Ubuntu 18.04.5 LTS             | Ubuntu 18.04.5 LTS             | Ubuntu 18.04.5 LTS             | Ubuntu 18.04.5 LTS             | Ubuntu 18.04.5 LTS             | Windows 11                                           |
| 处理器 | Intel(R) Xeon(R) CPU @ 2.20GHz | Intel(R) Xeon(R) CPU @ 2.20GHz | Intel(R) Xeon(R) CPU @ 2.20GHz | Intel(R) Xeon(R) CPU @ 2.20GHz | Intel(R) Xeon(R) CPU @ 2.20GHz | Intel(R) Core(TM) i5-10210U CPU @ 1.60GHz   2.11 GHz |

