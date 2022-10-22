import time
import numpy as np
from mindquantum import ParameterResolver
from mindspore import Tensor
from data_generator import circle_data_point_generator


def bench(n_qubits, epoch, batch, train_samples):

    Ntrain = train_samples
    Ntest = 100
    gap = 0.01

    train_x, train_y, test_x, test_y = circle_data_point_generator(
        Ntrain, Ntest, gap, n_qubits, 1
    )

    from mindquantum.core import Circuit

    # encoder
    encoder = Circuit()
    for i in range(n_qubits):
        encoder.ry(f"alpha_{i}", i)

    # ansatz
    ansatz = Circuit()
    for i in range(n_qubits):
        ansatz.rz(f"beta_1_{i}", i)
        ansatz.ry(f"beta_2_{i}", i)
        ansatz.rz(f"beta_3_{i}", i)

    for i in range(n_qubits - 1):
        ansatz.x(i + 1, i)
    ansatz.x(0, n_qubits - 1)

    for i in range(n_qubits):
        ansatz.ry(f"beta_4_{i}", i)
    circuit = encoder.as_encoder() + ansatz.as_ansatz()
    print(circuit)

    from mindquantum.core import QubitOperator
    from mindquantum.core import Hamiltonian

    # Hamiltonian
    hams = Hamiltonian(QubitOperator("Z0"))

    import mindspore as ms
    from mindquantum.framework import MQLayer
    from mindquantum.simulator import Simulator

    # model
    ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")
    ms.set_seed(1)
    sim = Simulator("mqvector_gpu", circuit.n_qubits)
    grad_ops = sim.get_expectation_with_grad(
        hams,
        circuit,
    )
    QuantumNet = MQLayer(grad_ops)

    from mindspore.nn import SoftmaxCrossEntropyWithLogits, L1Loss, MSELoss, LossBase
    from mindspore.nn import Adam, Accuracy, TrainOneStepCell
    from mindspore import Model
    from mindspore.dataset import (
        NumpySlicesDataset,
    )
    from mindspore.train.callback import (
        Callback,
        LossMonitor,
    )

    loss = MSELoss(reduction="mean")
    opti = Adam(QuantumNet.trainable_params(), learning_rate=0.1)
    model = Model(QuantumNet, loss, opti, metrics={"Acc": Accuracy()})

    train_loader = NumpySlicesDataset(
        {"features": train_x, "labels": train_y}, shuffle=False
    ).batch(batch)
    test_loader = NumpySlicesDataset({"features": test_x, "labels": test_y}).batch(
        batch
    )

    class StepAcc(Callback):
        def __init__(self, model, test_loader, train_loader):
            self.model = model
            self.test_loader = test_loader
            self.train_loader = train_loader
            self.acc = []

        def step_end(self, run_context):
            self.acc.append(
                self.model.eval(self.test_loader, dataset_sink_mode=False)["Acc"]
            )

    monitor = LossMonitor(5)

    acc = StepAcc(model, test_loader, train_loader)

    print("mindquantum: start training")
    # train
    start_time = time.time()
    model.train(epoch, train_loader, callbacks=[monitor, acc], dataset_sink_mode=False)
    end_time = time.time()
    print("mindquantum: finish")

    """
    # test acc
    weight = QuantumNet.weight.asnumpy()
    pr = dict(zip(ansatz.params_name, weight))
    circuit2 = encoder + ansatz.apply_value(pr)
    print(ansatz.apply_value(pr))
    sim2 = Simulator("projectq", circuit2.n_qubits)
    predict = []
    for i in range(len(test_x)):
        sim2.reset
        pr2 = dict(zip(encoder.params_name, test_x[i]))
        sim2.apply_circuit(circuit2, pr2)
        state_predict = (
            sim2.get_expectation(Hamiltonian(QubitOperator("Z0"))).real
        )
        if state_predict < 0:
            predict.append(-1)
        else:
            predict.append(1)
    acc = 0
    for i in range(len(test_y)):
        if test_y[i] == predict[i]:
            acc += 1
    print((len(test_y) - acc) / len(test_y))
    """

    return end_time - start_time


if __name__=='__main__':
    res = bench(4, 5, 20, 200)
    print(res)