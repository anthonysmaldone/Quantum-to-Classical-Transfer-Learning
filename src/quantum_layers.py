import math
import pennylane as qml

def AmplitudeQuantumLayer(input_data_per_circuit, vqc_layers=3):

    n_qubits = math.ceil(math.log2(input_data_per_circuit))+1
    dev = qml.device("default.qubit", wires=n_qubits)
    
    def _circuit(inputs, weights):
        qml.StatePrep(state=inputs, wires=range(1, n_qubits))
        qml.Hadamard(wires=[0])
        qml.ctrl(qml.adjoint(qml.StatePrep(state=inputs,wires=range(1,n_qubits))),control=[0])
        qml.ctrl(qml.StronglyEntanglingLayers(weights=weights, wires=range(1, n_qubits)),control=[0])
        qml.Hadamard(wires=[0])
        return qml.expval(qml.PauliZ(wires=0))
        
    qlayer = qml.QNode(_circuit, dev, interface="torch")
    weight_shapes = {"weights": (vqc_layers, n_qubits-1, 3)}
    return qml.qnn.TorchLayer(qml.transforms.broadcast_expand(qlayer), weight_shapes)
