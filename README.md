# Quantum-to-Classical-Transfer-Learning
![overall_flow_white](https://github.com/anthonysmaldone/Quantum-to-Classical-Transfer-Learning/assets/124306057/92d66e2c-533d-4fc5-97a9-754503c6ba50)

## Summary
We present a hybrid quantum-classical neural network for predicting drug toxicity, utilizing a novel quantum circuit design that mimics classical neural behavior by explicitly calculating matrix products with complexity $\mathcal{O}(n^2)$. Leveraging the Hadamard test for efficient inner product estimation rather than the conventionally used swap test, we reduce the number qubits by half and remove the need for quantum phase estimation. Directly computing matrix products quantum mechanically allows for learnable weights to be transferred from a quantum to a classical device for further training. We apply our framework to the Tox21 dataset and show that it achieves commensurate predictive accuracy to the model's fully classical analog. Additionally, we demonstrate the model continues to learn, without disruption, once transferred to a fully classical architecture. We believe combining the quantum advantage of reduced complexity and the classical advantage of noise-free calculation will pave the way to more scalable machine learning models.

## Pre-print
Coming soon

## Installation
Clone the Quantum-to-Classical-Transfer-Learning repository using:
```
git clone https://github.com/anthonysmaldone/Quantum-to-Classical-Transfer-Learning.git
```

Navigate to Quantum-to-Classical-Transfer-Learning with: 
```
cd Quantum-to-Classical-Transfer-Learning
```

Install the requirements with:
```
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
```

## Usage
Navigate to `src` with: 
```
cd src
```

See the training options by executing:
```
python train.py --h
```

## Example
To train the quantum neural network on the nr-er assay use:
```
python train.py --dataset nr-er
```

To train the fully classical analog, set the transfer epoch to 0:
```
python train.py --dataset nr-er --transfer_epoch 0
```


To train the hybrid model and derive the weights to continue training fully classical choose after which epoch the transfer occurs:
```
python train.py --dataset nr-er --transfer_epoch 5
```
