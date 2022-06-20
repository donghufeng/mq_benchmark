import tensorflow as tf
import tensorflow_quantum as tfq
from mindquantum.algorithm.nisq.qaoa import MaxCutAnsatz
import cirq

import sympy
import numpy as np
import pandas as pd


def trans_hamiltonian(mq_hamiltonion, qreg):
    gate_map = {
        "X": cirq.ops.X,
        "Y": cirq.ops.Y,
        "Z": cirq.ops.Z,
    }
    ham = cirq.PauliSum()
    for term in mq_hamiltonion.terms:
        coef = float(mq_hamiltonion.terms[term])  # for mindquantum==0.6.0
        # coef = float(mq_hamiltonion.terms[term].const)  # for mindquantum==0.6.2

        if len(term) == 0:
            ham += coef
            continue

        v = []
        for op in term:
            g = gate_map[op[1]]
            idx = int(op[0])
            v.append(g.on(qreg[idx]))

        ham += coef * cirq.PauliString(*tuple(v))

    return ham


def qaoa_circuit(graph, qreg, layer: int, pr_table):
    # Symbols for the rotation angles in the QAOA circuit.
    alpha = sympy.Symbol(f"alpha_{layer}")
    beta = sympy.Symbol(f"beta_{layer}")

    circ = cirq.Circuit(
        # Prepare uniform superposition on working_qubits == working_graph.nodes
        cirq.H.on_each(qreg),
        # Do ZZ operations between neighbors u, v in the graph.
        (
            cirq.ZZ(qreg[u], qreg[v]) ** alpha
            for (u, v) in graph.edges()
        ),
        # Apply X operations along all nodes of the graph.
        cirq.Moment(cirq.X(qubit) ** beta for qubit in qreg)
    )
    pr_table[f"alpha_{layer}"] = alpha
    pr_table[f"beta_{layer}"] = beta
    # print(pr_table)
    return circ


def bench(hyperparams={}):
    """
    Performs QAOA optimizations.

    Args:
            hyperparams (dict): hyperparameters to configure this benchmark

                    * 'graph': Graph represented as a NetworkX Graph class

                    * 'n_layers': Number of layers in the QAOA circuit

                    * 'shots': The number of samples.

                    * 'iter_num': The number of iterations
    """

    graph = hyperparams['graph']
    n_layers = hyperparams['n_layers']
    shots = hyperparams['shots']

    n_qubits = len(graph.nodes)
    qreg = cirq.LineQubit.range(n_qubits)

    # Construct the QAOA circuit
    total_circuit = cirq.Circuit()
    pr_table = dict()
    for i in range(n_layers):
        total_circuit.append(qaoa_circuit(graph, qreg, i, pr_table))

    # All relevant things can be computed in the computational basis.
    total_circuit.append(cirq.measure(qubit) for qubit in qreg)

    # 生成对应的的量子线路和哈密顿量
    maxcut = MaxCutAnsatz(list(graph.edges), n_layers)
    ham_mq = -maxcut.hamiltonian
    print("hamiltonian in mq:\n", ham_mq)

    # transform mindquantum hamiltonian to cirq
    ham = trans_hamiltonian(ham_mq, qreg)
    print("hamiltonian in cirq:\n", ham)

    print("circuit in mq:")
    print(maxcut.circuit)
    print("circuit in cirq:")
    print(total_circuit)

    expectation_calculation = tfq.layers.Expectation(
        differentiator=tfq.differentiators.ForwardDifference(grid_spacing=0.01)
    )

    theta = np.zeros((1, len(pr_table))).astype(np.float32)
    theta_tensor = tf.convert_to_tensor(theta)

    steps = hyperparams['iter_num']
    for i in range(steps):
        with tf.GradientTape() as g:
            g.watch(theta_tensor)
            output = expectation_calculation(
                total_circuit,
                operators=ham,
                symbol_names=list(pr_table.keys()),
                symbol_values=theta_tensor,
            )
            grad = g.gradient(output, theta_tensor)
            theta_tensor -= grad
            if (i + 1) % 10 == 0:
                print("training step:", i + 1, "  loss:", "%.4f" % output[0].numpy())

            # if i == steps - 1:
            #   print(output)
            #   print(theta_tensor)

    params = theta_tensor.numpy()
    print(params)

    pr_list = list(pr_table.keys())

    for i in range(len(pr_list)):
        pr_table[pr_list[i]] = params[0][i]

    print(pr_table)

    sim = cirq.Simulator()
    sample_results = sim.sample(total_circuit, params=pr_table, repetitions=shots)

    # Results statistics
    sample_results = np.array(sample_results)
    # print(sample_results)

    results = dict()
    for sample in sample_results:
        sample_str = ""
        for q in sample[-n_qubits:]:
            sample_str += str(int(q))

        if sample_str not in results:
            results[sample_str] = 1
        else:
            results[sample_str] += 1

    print("Measurement result with Tensorflow Quantum:\n", results)
