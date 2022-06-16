import os
os.environ["OMP_NUMBER_THREADS"] = "3"

import time

from pennylane import numpy as np 
import pennylane as qml
from pennylane import qchem
from functools import partial


def info_from_data(data):
    symbols = []
    coordinates = []
    for par in data:
        atom, coordinate = par[0], par[1]
        symbols.append(atom)
        for x in coordinate:
            coordinates.append(x)
    return (symbols, np.array(coordinates))

def get_electrons(atoms) -> int:
    ans = 0 
    counter = {
        "H" : 1, 
        "He" : 2, 
        "Li" : 3, 
        "Be" : 4,
        "B" : 5,
        "C" : 6,
        "N" : 7,
        "O" : 8,
        "F" : 9,
        "Ne" : 10,
        "Na" : 11,
        "Mg" : 12,
        "Al" : 13,
        "Si" : 14,
        "P" : 15,
        "S" : 16,
        "Cl" : 17,
        "Ar" : 18,
        "K" : 19,
        "Ga" : 20,
    }
    for a in atoms:
        if a not in counter:
            raise ValueError(f"Invalid atom {a}.")
        ans += counter[a]
    return ans

def bench(data, iter_num):
    symbols, coordinates = info_from_data(data)
    electrons = get_electrons(symbols)
    print(symbols)
    print(coordinates)
    print(f"electrons = {electrons}")

    H, qubits = qml.qchem.molecular_hamiltonian(symbols, coordinates)
    print("Number of qubits = ", qubits)
    # print(f"The Hamiltonian is {H}")

    hf = qml.qchem.hf_state(electrons, qubits)
    print(f"hf = {hf}")

    dev = qml.device("default.qubit", wires=qubits)

    # Generate single and double excitations
    singles, doubles = qchem.excitations(electrons, qubits)

    # Map excitations to the wires the UCCSD circuit will act on
    s_wires, d_wires = qchem.excitations_to_wires(singles, doubles)

    # Define the UCCSD ansatz
    ansatz = partial(qml.UCCSD, init_state=hf, s_wires=s_wires, d_wires=d_wires)

    # Define the cost function
    cost_fn = qml.ExpvalCost(ansatz, H, dev)

    opt = qml.GradientDescentOptimizer(stepsize=0.4)

    # theta = np.array(0.0, requires_grad=True)
    theta = np.random.normal(0, np.pi, len(singles)+len(doubles), requires_grad=True)

    energy = [cost_fn(theta)]
    angle = [theta]

    max_iterations = iter_num
    conv_tol = 1e-6

    start_time = time.time()

    for n in range(max_iterations):
        theta, prev_energy = opt.step_and_cost(cost_fn, theta)

        energy.append(cost_fn(theta))
        angle.append(theta)

        conv = np.abs(energy[-1] - prev_energy)

        if n % 5 == 0:
            print(f"Step = {n}, Energy = {energy[-1]:.8f} Ha")
    
    end_time = time.time()

    print(f"Final value of the ground-state energy = {energy[-1]:.8f} Ha")
    print(f"Optimal value of the circuit parameter = {angle[-1]}")

    print(f"Used time: {end_time - start_time}")
    return end_time - start_time

if __name__ == "__main__":
    data = [["H", [0.0, 0.0, -0.6614]], ["H", [0.0, 0.0, 0.6614]]]
    iter_num = 50
    bench(data, iter_num)
