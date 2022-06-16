import os
os.environ["OMP_NUMBER_THREADS"] = "3"

import pyqpanda as pq 
import numpy as np

from openfermion.chem import MolecularData
from openfermionpyscf import run_pyscf
import mindquantum as mq

import time

def parse_braces(w: str, lb: str, rb: str) -> str:
    i = w.find(lb)
    j = w.find(rb)
    return w[i+1:j]

def trans_hamiltonian(mq_ham):
    ops = dict()
    for l in str(mq_ham).splitlines():
        op = parse_braces(l, "[", "]")
        idx = l.find("[")
        x = float(l[:idx])
        ops[op] = x
    return pq.PauliOperator(ops)

from mindquantum.core import gates as mgates
from mindquantum.core import Circuit as mcircuit

def trans_circuit_mindquantum_qpanda(circuit: mcircuit, n_qubits: int, machine, q):
    vqc = pq.VariationalQuantumCircuit()

    def self_herm_non_params(gate):
        ctrls = gate.ctrl_qubits
        objs = gate.obj_qubits
        if ctrls:
            # must be CNOT
            g = pq.VariationalQuantumGate_CNOT
            g = g(q[ctrls[0]], q[objs[0]])
            vqc.insert(g)
        else:
            # must be H
            gate_map = {
                "X": pq.VariationalQuantumGate_X,
                "Y": pq.VariationalQuantumGate_Y,
                "Z": pq.VariationalQuantumGate_Z,
                "H": pq.VariationalQuantumGate_H,
            }
            g = gate_map[gate.name.upper()]
            g = g(q[objs[0]])
            vqc.insert(g)
    
    def params_gate_trans(gate, pr_table):
        gate_map = {
            "RX": pq.VariationalQuantumGate_RX,
            "RY": pq.VariationalQuantumGate_RY,
            "RZ": pq.VariationalQuantumGate_RZ,
        }
        objs = gate.obj_qubits
        if gate.ctrl_qubits:
            raise ValueError(f"Can't convert {gate} with params.")
        g = gate_map[gate.name.upper()]
        if gate.parameterized:
            # parameter
            acc = None
            for k,v in gate.coeff.items():
                if k not in pr_table:
                    pr_table[k] = pq.var(1, True)
                if acc is None:
                    acc = v * pr_table[k]
                else:
                    acc += v * pr_table[k]
            g = g(q[objs[0]], acc)
        else:
            # no parameter
            g = g(q[objs[0]], gate.coeff.const)
        vqc.insert(g)

    cnt1, cnt2 = 0, 0
    circuit = circuit.remove_barrier()
    pr_table = dict()
    for g in circuit:
        if isinstance(g, (
            mgates.XGate, mgates.HGate
        )):
            cnt1 += 1
            self_herm_non_params(g)
        elif isinstance(g, (
            mgates.RX, mgates.RY, mgates.RZ
        )):
            cnt2 += 1
            params_gate_trans(g, pr_table)
        else:
            raise ValueError(f"Haven't implemented convertion for gate {g}")
    print(f"cnt1={cnt1}, cnt2={cnt2}")
    return vqc


def bench(data, iter_num):
    molecule_of = MolecularData(
        geometry=data, 
        basis="sto3g", 
        multiplicity=1
    )
    molecule_of = run_pyscf(
        molecule_of, 
        run_scf=1, 
        run_ccsd=1, 
        run_fci=1
    )
    molecule_of.save()
    molecule_file = molecule_of.filename
    hartreefock_wfn_circuit = mq.Circuit([
        mq.X.on(i) for i in range(molecule_of.n_electrons)
    ])
    ansatz_circuit, \
    init_amplitudes, \
    ansatz_parameter_names, \
    hamiltonian_QubitOp, \
    n_qubits, \
    n_electrons = mq.algorithm.generate_uccsd(molecule_file, th=-1)

    total_circuit = hartreefock_wfn_circuit + ansatz_circuit
    total_circuit.summary()
    print("Number of parameters: %d" % (len(ansatz_parameter_names)))

    # qpanda
    machine = pq.init_quantum_machine(pq.QMachineType.CPU)
    qubit_list = machine.qAlloc_many(n_qubits)

    vqc = trans_circuit_mindquantum_qpanda(
        total_circuit, n_qubits, machine, qubit_list)

    ham = trans_hamiltonian(hamiltonian_QubitOp)

    loss = pq.qop(vqc, ham, machine, qubit_list)

    optimizer = pq.MomentumOptimizer.minimize(loss, 0.05, 0.9)

    leaves = optimizer.get_variables()

    print("Start training.")
    start_time = time.time()
    for i in range(iter_num):
        optimizer.run(leaves, 0)
        loss_value = optimizer.get_loss()
        if i % 5 == 0:
            print(f"step {i}: loss = {loss_value}")
    end_time = time.time()
    print(f"Used time: {end_time-start_time}")
    return end_time - start_time


if __name__ == "__main__":
    # data = [["H", [0.0, 0.0, -0.6614]], ["H", [0.0, 0.0, 0.6614]]]
    data = [["Li", [0, 0, 0]], ["H", [1, 0, 0]]]
    iter_num = 100

    bench(data, iter_num)
