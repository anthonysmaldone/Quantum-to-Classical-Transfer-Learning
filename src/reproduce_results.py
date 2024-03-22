import os

commands = [
    'python train.py --dataset nr-ahr --device cuda --notes nr-ahr-quantum',
    'python train.py --dataset nr-ahr --device cuda --transfer_epoch 0 --notes nr-ahr-classical',
    'python train.py --dataset nr-ahr --device cuda --transfer_epoch 5 --notes nr-ahr-transfer',
    'python train.py --dataset nr-ar --device cuda --notes nr-ar-quantum',
    'python train.py --dataset nr-ar --device cuda --transfer_epoch 0 --notes nr-ar-classical',
    'python train.py --dataset nr-ar --device cuda --transfer_epoch 5 --notes nr-ar-transfer',
    'python train.py --dataset nr-ar-lbd --device cuda --notes nr-ar-lbd-quantum',
    'python train.py --dataset nr-ar-lbd --device cuda --transfer_epoch 0 --notes nr-ar-lbd-classical',
    'python train.py --dataset nr-ar-lbd --device cuda --transfer_epoch 5 --notes nr-ar-lbd-transfer',
    'python train.py --dataset nr-aromatase --device cuda --notes nr-aromatase-quantum',
    'python train.py --dataset nr-aromatase --device cuda --transfer_epoch 0 --notes nr-aromatase-classical',
    'python train.py --dataset nr-aromatase --device cuda --transfer_epoch 5 --notes nr-aromatase-transfer',
    'python train.py --dataset nr-er --device cuda --notes nr-er-quantum',
    'python train.py --dataset nr-er --device cuda --transfer_epoch 0 --notes nr-er-classical',
    'python train.py --dataset nr-er --device cuda --transfer_epoch 5 --notes nr-er-transfer',
    'python train.py --dataset nr-er-lbd --device cuda --notes nr-er-lbd-quantum',
    'python train.py --dataset nr-er-lbd --device cuda --transfer_epoch 0 --notes nr-er-lbd-classical',
    'python train.py --dataset nr-er-lbd --device cuda --transfer_epoch 5 --notes nr-er-lbd-transfer',
    'python train.py --dataset nr-ppar-gamma --device cuda --notes nr-ppar-gamma-quantum',
    'python train.py --dataset nr-ppar-gamma --device cuda --transfer_epoch 0 --notes nr-ppar-gamma-classical',
    'python train.py --dataset nr-ppar-gamma --device cuda --transfer_epoch 5 --notes nr-ppar-gamma-transfer',
    'python train.py --dataset sr-are --device cuda --notes sr-are-quantum',
    'python train.py --dataset sr-are --device cuda --transfer_epoch 0 --notes sr-are-classical',
    'python train.py --dataset sr-are --device cuda --transfer_epoch 5 --notes sr-are-transfer',
    'python train.py --dataset sr-atad5 --device cuda --notes sr-atad5-quantum',
    'python train.py --dataset sr-atad5 --device cuda --transfer_epoch 0 --notes sr-atad5-classical',
    'python train.py --dataset sr-atad5 --device cuda --transfer_epoch 5 --notes sr-atad5-transfer',
    'python train.py --dataset sr-hse --device cuda --notes sr-hse-quantum',
    'python train.py --dataset sr-hse --device cuda --transfer_epoch 0 --notes sr-hse-classical',
    'python train.py --dataset sr-hse --device cuda --transfer_epoch 5 --notes sr-hse-transfer',
    'python train.py --dataset sr-mmp --device cuda --notes sr-mmp-quantum',
    'python train.py --dataset sr-mmp --device cuda --transfer_epoch 0 --notes sr-mmp-classical',
    'python train.py --dataset sr-mmp --device cuda --transfer_epoch 5 --notes sr-mmp-transfer',
    'python train.py --dataset sr-p53 --device cuda --notes sr-p53-quantum',
    'python train.py --dataset sr-p53 --device cuda --transfer_epoch 0 --notes sr-p53-classical',
    'python train.py --dataset sr-p53 --device cuda --transfer_epoch 5 --notes sr-p53-transfer',
    'python train.py --dataset nr-ahr --device cuda --transfer_epoch 5 --notes nr-ahr_ablation --ablation True '
]

for cmd in commands:
    print(f"Running command: {cmd}")
    os.system(cmd)
    print("----------------------------------------------------")
