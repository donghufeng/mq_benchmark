import os

os.environ["OMP_NUMBER_THREADS"] = "3"

import time

import tensorflow as tf
import tensorflow_quantum as tfq

import cirq
import sympy
import numpy as np

import matplotlib.pyplot as plt
from cirq.contrib.svg import SVGCircuit

from openfermion.chem import MolecularData
from openfermionpyscf import run_pyscf
import mindquantum as mq


def trans_hamiltonian(mq_hamiltonion, qreg):
    gate_map = {
        "X": cirq.ops.X,
        "Y": cirq.ops.Y,
        "Z": cirq.ops.Z,
    }
    ham = cirq.PauliSum()
    for term, coeff in mq_hamiltonion.terms:
        ham += coeff.const * cirq.PauliString(
            *tuple([gate_map[p].on(idx) for idx, p in term.items()]))
    return ham


from mindquantum.core import gates as mgates
from mindquantum.core import Circuit as mcircuit


def trans_circuit_mindquantum_cirq(mcircuit: mcircuit, n_qubits: int, qreg,
                                   pr_table):
    circ = cirq.Circuit()

    def self_herm_non_params(gate):
        ctrls = gate.ctrl_qubits
        objs = gate.obj_qubits
        if ctrls:
            # must be CNOT
            circ.append([cirq.ops.CNOT.on(qreg[ctrls[0]], qreg[objs[0]])])
        else:
            # must be H
            gate_map = {
                "X": cirq.ops.X,
                "Y": cirq.ops.Y,
                "Z": cirq.ops.Z,
                "H": cirq.ops.H,
            }
            g = gate_map[gate.name.upper()]
            circ.append([g.on(qreg[objs[0]])])

    def params_gate_trans(gate, pr_table):
        gate_map = {
            "RX": cirq.ops.rx,
            "RY": cirq.ops.ry,
            "RZ": cirq.ops.rz,
        }
        objs = gate.obj_qubits
        if gate.ctrl_qubits:
            raise ValueError(f"Can't convert {gate} with params.")

        g = gate_map[gate.name.upper()]
        if gate.parameterized:
            # parameter
            # tfq can't support Rx(alpha + beta),
            # so have to convert to Rx(alpha)Rx(beta)
            for k, v in gate.coeff.items():
                if k not in pr_table:
                    pr_table[k] = sympy.Symbol(k)
                circ.append([g(v * pr_table[k]).on(qreg[objs[0]])])
        else:
            # no parameter
            g = g(gate.coeff.const).on(qreg[objs[0]])
            circ.append([g])

    cnt1, cnt2 = 0, 0
    mcircuit = mcircuit.remove_barrier()
    # pr_table = dict()
    for g in mcircuit:
        if isinstance(g, (mgates.XGate, mgates.HGate)):
            cnt1 += 1
            self_herm_non_params(g)
        elif isinstance(g, (mgates.RX, mgates.RY, mgates.RZ)):
            cnt2 += 1
            params_gate_trans(g, pr_table)
        else:
            raise ValueError(f"Haven't implemented convertion for gate {g}")
    print(f"cnt1={cnt1}, cnt2={cnt2}")
    return circ


def bench(data, iter_num):
    molecule_of = MolecularData(geometry=data, basis="sto3g", multiplicity=1)

    molecule_of = run_pyscf(molecule_of, run_scf=1, run_ccsd=1, run_fci=1)
    molecule_of.save()
    molecule_file = molecule_of.filename
    # print(molecule_file)

    hartreefock_wfn_circuit = mq.Circuit(
        [mq.X.on(i) for i in range(molecule_of.n_electrons)])
    print(hartreefock_wfn_circuit)

    ansatz_circuit, \
    init_amplitudes, \
    ansatz_parameter_names, \
    hamiltonian_QubitOp, \
    n_qubits, \
    n_electrons = mq.algorithm.generate_uccsd(molecule_file, th=-1)

    total_circuit = hartreefock_wfn_circuit + ansatz_circuit
    total_circuit.summary()
    print("Number of parameters: %d" % (len(ansatz_parameter_names)))

    # transform mindquantum to cirq
    qreg = cirq.LineQubit.range(n_qubits)
    ham = trans_hamiltonian(hamiltonian_QubitOp, qreg)
    pr_table = dict()
    circ = trans_circuit_mindquantum_cirq(total_circuit, n_qubits, qreg,
                                          pr_table)

    expectation_calculation = tfq.layers.Expectation(
        differentiator=tfq.differentiators.Adjoint())

    theta = np.zeros((1, len(pr_table))).astype(np.float32)
    theta_tensor = tf.convert_to_tensor(theta)
    print(theta_tensor.shape)

    print(f"Start training.")
    start_time = time.time()
    for i in range(iter_num):
        with tf.GradientTape() as g:
            g.watch(theta_tensor)
            output = expectation_calculation(
                circ,
                operators=ham,
                symbol_names=list(pr_table.keys()),
                symbol_values=theta_tensor,
            )
            grad = g.gradient(output, theta_tensor)
            theta_tensor -= grad
            if i % 1 == 0:
                print(f"Step {i}: loss = {output[0]}")
    end_time = time.time()
    print(f"Used time: {end_time - start_time}")


if __name__ == "__main__":
    # data = [["H", [0.0, 0.0, -0.6614]], ["H", [0.0, 0.0, 0.6614]]]
    data = [["Li", [0, 0, 0]], ["H", [1, 0, 0]]]
    iter_num = 50
    bench(data, iter_num)