# Quantum-to-Classical-Transfer-Learning
![github_repo_img](https://github.com/anthonysmaldone/Quantum-to-Classical-Transfer-Learning/assets/124306057/2818bd94-b711-42d0-8199-159f2c75888a)

## Summary
Toxicity is a roadblock that prevents an inordinate number of drugs from being used in potentially life-saving applications. Deep learning provides a promising solution to finding ideal drug candidates; however, the vastness of chemical space coupled with the underlying $\mathcal{O}(n^3)$ matrix multiplication means these efforts quickly become computationally demanding. To remedy this, we present a hybrid quantum-classical neural network for predicting drug toxicity, utilizing a quantum circuit design that mimics classical neural behavior by explicitly calculating matrix products with complexity $\mathcal{O}(n^2)$. Leveraging the Hadamard test for efficient inner product estimation rather than the conventionally used swap test, we reduce the number qubits by half and remove the need for quantum phase estimation. Directly computing matrix products quantum mechanically allows for learnable weights to be transferred from a quantum to a classical device for further training. We apply our framework to the Tox21 dataset and show that it achieves commensurate predictive accuracy to the model's fully classical $\mathcal{O}(n^3)$ analog. Additionally, we demonstrate the model continues to learn, without disruption, once transferred to a fully classical architecture. We believe combining the quantum advantage of reduced complexity and the classical advantage of noise-free calculation will pave the way to more scalable machine learning models. 

## Publication
[https://pubs.acs.org/doi/abs/10.1021/acs.jctc.4c00432](https://pubs.acs.org/doi/abs/10.1021/acs.jctc.4c00432)

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

## Reproduce Paper Experiments
To exactly reproduce all the works in this repository's corresponding research article, run the following command:
```
python reproduce_results.py
```
