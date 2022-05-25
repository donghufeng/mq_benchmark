import sys
import benchmark_paddlepaddle
import benchmark_pennylane

args = sys.argv
if len(args) == 1:
    print("Please input the number of qubits.")
elif len(args) > 2:
    print("Please input only one number.")
else:
    n_qubits = args[1]
    print(f"n_qubits={n_qubits}")

    epoch = 5
    batch = 20
    train_samples = 200

    use_time = benchmark_paddlepaddle.bench(n_qubits, epoch, batch, train_samples)
