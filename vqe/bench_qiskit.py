import os
os.environ["OMP_NUMBER_THREADS"] = "3"

import time

from qiskit import Aer
from qiskit_nature.drivers import UnitsType, Molecule
from qiskit_nature.drivers.second_quantization import (
    ElectronicStructureDriverType,
    ElectronicStructureMoleculeDriver,
)
from qiskit_nature.problems.second_quantization import ElectronicStructureProblem
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper

from qiskit.providers.aer import StatevectorSimulator
from qiskit.utils import QuantumInstance
from qiskit_nature.algorithms import VQEUCCFactory

from qiskit_nature.algorithms import GroundStateEigensolver

def bench(data):
    molecule = Molecule(
        geometry=data, 
        charge=0, 
        multiplicity=1
    )
    driver = ElectronicStructureMoleculeDriver(
        molecule, basis="sto3g", driver_type=ElectronicStructureDriverType.PYSCF
    )

    es_problem = ElectronicStructureProblem(driver)
    qubit_converter = QubitConverter(JordanWignerMapper())

    quantum_instance = QuantumInstance(backend=Aer.get_backend("aer_simulator_statevector"))
    vqe_solver = VQEUCCFactory(quantum_instance)

    calc = GroundStateEigensolver(qubit_converter, vqe_solver)

    print("Start solving.")
    start_time = time.time()
    res = calc.solve(es_problem)
    end_time = time.time()

    print(res)
    print(f"Used time: {end_time-start_time}")
    return end_time - start_time


if __name__ == "__main__":
    data = [["H", [0.0, 0.0, 0.0]], ["H", [0.0, 0.0, 0.735]]]

    bench(data)
