import torch
from collections import OrderedDict
import pennylane as qml


def computed_weights(weights):
    def circuit(weight):
        qml.StronglyEntanglingLayers(weights=weight, wires=range(weight.shape[1]))
        return qml.state()
    return torch.real(qml.matrix(circuit)(weights)[:,0])

def save_quantum_weight_file(quantum_model_path):

    pre_loaded_quantum_model = torch.load(quantum_model_path)

    weights_list = []

    for key in pre_loaded_quantum_model.keys():
        # Check if the key matches the pattern that follows Pennylane's state dict naming convention
        if "quantum_layer.QuantumLayer" in key and ".weights" in key:
            # Extract the weights and add them to the list
            weights = pre_loaded_quantum_model[key]
            weights_list.append(weights)


    weights_list = torch.stack(weights_list)[0]
    
    # Ensure the shape matches the state dict shape for PyTorchs conv filters
    transfer_conv = computed_weights(weights_list).reshape((1,1,2,2))

    new_state_dict = OrderedDict()

    new_state_dict['conv_layer.weight'] = transfer_conv
    new_state_dict['conv_layer.bias'] = pre_loaded_quantum_model['quantum_layer.QuantumLayerBias']
    new_state_dict['conv_layer2.weight'] = pre_loaded_quantum_model['conv_layer2.weight']
    new_state_dict['conv_layer2.bias'] = pre_loaded_quantum_model['conv_layer2.bias']
    new_state_dict['fc1.weight'] = pre_loaded_quantum_model['fc1.weight']
    new_state_dict['fc1.bias'] = pre_loaded_quantum_model['fc1.bias']

    print("Saving extracted weights from quantum system")
    torch.save(new_state_dict, "./transferred_weights.pth")
