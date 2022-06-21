import cirq
import sympy
from mindquantum.core import gates as mgates
from mindquantum.core import Circuit as mcircuit
import pyqpanda as pq
from qiskit import QuantumCircuit


def parse_braces(w: str, lb: str, rb: str) -> str:
    i = w.find(lb)
    j = w.find(rb)
    return w[i + 1 : j]


class mq_to_tfq:
    def trans_hamiltonian(mq_hamiltonion, qreg):
        gate_map = {
            "X": cirq.ops.X,
            "Y": cirq.ops.Y,
            "Z": cirq.ops.Z,
        }
        ham = cirq.PauliSum()
        for l in str(mq_hamiltonion).splitlines():
            op = parse_braces(l, "[", "]")
            idx = l.find("[")
            x = float(l[:idx])

            if op == "":
                ham += x
                continue

            v = []
            for w in op.split(" "):
                g = gate_map[w[0]]
                idx = int(w[1:])
                v.append(g.on(qreg[idx]))

            ham += x * cirq.PauliString(*tuple(v))
        return ham

    def trans_circuit_mindquantum_cirq(
        mcircuit: mcircuit, n_qubits: int, qreg, pr_table
    ):
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


class mq_to_qpanda:
    def trans_hamiltonian(mq_ham):
        ops = dict()
        for l in str(mq_ham).splitlines():
            op = parse_braces(l, "[", "]")
            idx = l.find("[")
            x = float(l[:idx])
            ops[op] = x
        return pq.PauliOperator(ops)

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
                for k, v in gate.coeff.items():
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
            if isinstance(g, (mgates.XGate, mgates.HGate)):
                cnt1 += 1
                self_herm_non_params(g)
            elif isinstance(g, (mgates.RX, mgates.RY, mgates.RZ)):
                cnt2 += 1
                params_gate_trans(g, pr_table)
            else:
                raise ValueError(f"Haven't implemented convertion for gate {g}")
        print(f"cnt1={cnt1}, cnt2={cnt2}")
        return vqc, pr_table


def params_trans(pr, pr_table, to_qiskit=True):
    from qiskit.circuit import Parameter

    if to_qiskit:
        out = None
        for k, v in pr.items():
            if k not in pr_table:
                pr_table[k] = Parameter(k)
            if out is None:
                out = pr_table[k] * v
            else:
                out += pr_table[k] * v
        out += pr.const
        return out


def self_herm_non_params(gate, circ, to_qiskit=True):
    from qiskit.circuit import library as qlib

    qgate_map = {
        "X": qlib.XGate,
        "Y": qlib.YGate,
        "Z": qlib.ZGate,
        "H": qlib.HGate,
        "SWAP": qlib.SwapGate,
        "ISWAP": qlib.iSwapGate,
    }
    if to_qiskit:
        ctrls = gate.ctrl_qubits
        objs = gate.obj_qubits
        g = qgate_map[gate.name.upper()]()
        if ctrls:
            g = g.control(len(ctrls))
        circ.append(g, ctrls + objs, [])


def t_gate_trans(gate, circ, to_qiskit=True):
    from qiskit.circuit import library as qlib

    if to_qiskit:
        ctrls = gate.ctrl_qubits
        objs = gate.obj_qubits
        if gate.hermitianed:
            g = qlib.TdgGate()
        else:
            g = qlib.TGate()
        if ctrls:
            g = g.control(len(ctrls))
        circ.append(g, ctrls + objs, [])


def s_gate_trans(gate, circ, to_qiskit=True):
    from qiskit.circuit import library as qlib

    if to_qiskit:
        ctrls = gate.ctrl_qubits
        objs = gate.obj_qubits
        if gate.hermitianed:
            g = qlib.SdgGate()
        else:
            g = qlib.SGate()
        if ctrls:
            g = g.control(len(ctrls))
        circ.append(g, ctrls + objs, [])


def oppo_params_gate_trans(gate, circ, pr_table, to_qiskit=True):
    from qiskit.circuit import library as qlib

    qgate_map = {
        "RX": qlib.RXGate,
        "RY": qlib.RYGate,
        "RZ": qlib.RZGate,
        "ZZ": qlib.RZZGate,
        "YY": qlib.RYYGate,
        "XX": qlib.RXXGate,
        "PS": qlib.PhaseGate,
    }
    if to_qiskit:
        ctrls = gate.ctrl_qubits
        objs = gate.obj_qubits
        if gate.parameterized:
            g = qgate_map[gate.name.upper()](params_trans(gate.coeff, pr_table))
        else:
            g = qgate_map[gate.name.upper()](gate.coeff.const)
        if ctrls:
            g = g.control(len(ctrls))
        circ.append(g, ctrls + objs, [])


def to_qiskit(circuit: mcircuit):
    qcircuit = QuantumCircuit(circuit.n_qubits)
    pr_table = {}
    for g in circuit:
        if isinstance(
            g, (mgates.XGate, mgates.YGate, mgates.ZGate, mgates.HGate, mgates.SWAPGate)
        ):
            self_herm_non_params(g, qcircuit)
        elif isinstance(g, mgates.TGate):
            t_gate_trans(g, qcircuit)
        elif isinstance(g, mgates.SGate):
            t_gate_trans(g, qcircuit)
        elif isinstance(
            g,
            (
                mgates.RX,
                mgates.RY,
                mgates.RZ,
                mgates.ZZ,
                mgates.YY,
                mgates.XX,
                mgates.PhaseShift,
            ),
        ):
            oppo_params_gate_trans(g, qcircuit, pr_table)
        else:
            raise ValueError(f"Do not know how to convert {g} to qiskit.")
    return qcircuit
