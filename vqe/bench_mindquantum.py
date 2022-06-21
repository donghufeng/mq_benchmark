import os
# os.environ["OMP_NUMBER_THREADS"] = "3"

import time

import numpy as np
from openfermion.chem import MolecularData
from openfermionpyscf import run_pyscf
import mindquantum as mq
from mindquantum import Circuit, X, RX, Hamiltonian, Simulator
from mindquantum.algorithm import generate_uccsd
import mindspore as ms
import mindspore.context as context
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer
from mindquantum.framework import MQAnsatzOnlyLayer

context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")

def bench(data, iter_num: int):
    geometry = data
    basis = "sto3g"
    spin = 0
    # print("Geometry: ", geometry)

    molecule_of = MolecularData(
        geometry, 
        basis, 
        multiplicity=2*spin+1
    )

    molecule_of = run_pyscf(
        molecule_of, 
        run_scf=1, 
        run_ccsd=1, 
        run_fci=1
    )

    # print("Hartree-Fock energy: %20.16f Ha" % (molecule_of.hf_energy))
    # print("CCSD energy: %20.16f Ha" % (molecule_of.ccsd_energy))
    # print("FCI  energy: %20.16f Ha" % (molecule_of.fci_energy))

    molecule_of.save()
    molecule_file = molecule_of.filename
    # print(molecule_file)

    hartreefock_wfn_circuit = Circuit([
        X.on(i) for i in range(molecule_of.n_electrons)
    ])
    # print(hartreefock_wfn_circuit)

    ansatz_circuit, \
    init_amplitudes, \
    ansatz_parameter_names, \
    hamiltonian_QubitOp, \
    n_qubits, \
    n_electrons = generate_uccsd(molecule_file, th=-1)

    total_circuit = hartreefock_wfn_circuit + ansatz_circuit
    total_circuit.summary()
    print("Number of parameters: %d" % (len(ansatz_parameter_names)))

    sim = Simulator("projectq", total_circuit.n_qubits)
    ham = Hamiltonian(hamiltonian_QubitOp)

    # todo: 程序闪退
    ham.sparse(total_circuit.n_qubits)
    
    molecule_pqc = sim.get_expectation_with_grad(
        ham,
        total_circuit
    )

    molecule_pqcnet = MQAnsatzOnlyLayer(molecule_pqc, "Zeros")

    initial_energy = molecule_pqcnet()
    # print("Initial energy: %20.16f" % (initial_energy.asnumpy()))

    optimizer = ms.nn.Adagrad(
        molecule_pqcnet.trainable_params(),
        learning_rate=4e-2
    )
    train_pqcnet = ms.nn.TrainOneStepCell(molecule_pqcnet, optimizer)

    eps = 1.e-8
    energy_diff = eps * 1000
    energy_last = initial_energy.asnumpy() + energy_diff
    iter_idx = 0

    # time it
    # limit iteration
    print("Start training.")
    start_time = time.time()

    # while abs(energy_diff) > eps:
    while iter_idx < iter_num:
        energy_i = train_pqcnet().asnumpy()
        if iter_idx % 5 == 0:
            print("Step %3d energy %20.16f" % (iter_idx, float(energy_i)))
        energy_diff = energy_last - energy_i
        energy_last = energy_i
        iter_idx += 1

    end_time = time.time()
    print(f"Time of training: {end_time - start_time}")

    # print("Optimization completed at step %3d" % (iter_idx - 1))
    # print("Optimized energy: %20.16f" % energy_i)
    # print("Optimized amplitudes: ", molecule_pqcnet.weight.asnumpy())

    return end_time - start_time

if __name__ == "__main__":
    data = [["C", [0, 0, 0]], ["H", [0.6276, 0.6276, 0.6276]], ["H", [0.6276, -0.6276, -0.6276]], ["H", [-0.6276, 0.6276, 0.6276]], ["H", [-0.6276, -0.6276, 0.6276]]]
    iter_num = 50
    bench(data, iter_num=iter_num)
