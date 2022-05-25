# QNN

## Usage

- 根据需要修改 `bench_qnn.py`，终端运行命令 `python bench_qnn.py n_qubits`，其中`n_qubits`为比特数，用来调整消耗的资源规模。
- 各个框架的代码位于 `benchmark_NAME.py` 下，其中 `NAME` 为框架名，例如：`mindquantum`、`paddlepaddle`。
- `data_generator.py`用于生成圆形决策边界两分类数据集，`tran_mcircuit.py`有部分由mindquantum线路转化成其它框架线路的方法。