import torch
import torch.nn as nn
import torch.optim as optim
import os
import models
import prepare_data
import log
import argparse
import weight_extraction
from sklearn.metrics import roc_auc_score
import torch.nn.init as init

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model on the specified dataset.")
    parser.add_argument('--dataset', type=str, default='nr-ahr', help='Choose Tox21 experiment: nr-ahr, nr-ar, nr-ar-lbd, nr-armoatase, nr-er, nr-er-lbd, nr-ppar-gamma ,sr-are, sr-atad5, sr-hse, sr-mmp, sr-p53')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--num_epochs', type=int, default=10, help='Total number of epochs')
    parser.add_argument('--notes', type=str, default="", help='Optional Notes')
    parser.add_argument('--device', type=str, default="cpu", help='cpu or cuda')
    parser.add_argument('--quantum_model_file', type=str, default="None", help='Path to quantum model to resume training')
    parser.add_argument('--transfer_epoch', type=int, default=-1, help='Epoch when quantum transfers to classical (0 for all classical, -1 for all quantum)')
    parser.add_argument('--seed', type=int, default=123, help='Set the torch.manual_seed')
    
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    dataset = args.dataset
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    notes = args.notes
    device = args.device
    transfer_epoch = args.transfer_epoch
    quantum_model_path = args.quantum_model_file
    if args.quantum_model_file != "None":
        quantum_model_path = os.path.abspath(os.path.normpath(args.quantum_model_file))
    seed = args.seed

    hyperparameters = {
        "dataset": dataset,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "notes": notes,
        "seed": seed
    }
    
    print(
        f"Training on dataset: {dataset}, "
        f"batch size: {batch_size}, "
        f"learning rate: {learning_rate}, "
        f"total number of epochs: {num_epochs}, "
        f"notes: {notes}, "
        f"device: {device}"
    )

    log_file_path, current_datetime, model_save_dir = log.prepare_log(hyperparameters)
    train_dataloader, test_dataloader = prepare_data.prepare_dataloaders(batch_size=batch_size, dataset=dataset, seed=seed)
    
    quantum_model = models.QuantumCNN(dataset=dataset, out_channels=1, in_channels=1, vqc_layers=3, device=device)

    criterion = nn.BCEWithLogitsLoss()
    
    if transfer_epoch == 0 and quantum_model_path == "None":
        classical_model = models.ClassicalCNN(dataset=dataset, in_channels=1, out_channels=1, kernel_size=2, device=device)
        
        # Initialize both models with the same values to ensure fair comparison
        classical_model.fc1.weight.data = quantum_model.fc1.weight.data.clone()
        classical_model.fc1.bias.data = quantum_model.fc1.bias.data.clone()
        classical_model.conv_layer2.weight.data = quantum_model.conv_layer2.weight.data.clone()
        classical_model.conv_layer2.bias.data = quantum_model.conv_layer2.bias.data.clone()
    
    elif transfer_epoch == 0 and quantum_model_path != "None":
        classical_model = models.ClassicalCNN(dataset=dataset, in_channels=1, out_channels=1, kernel_size=2, transfer=True, device=device)
        
        # Derive the classical weights from the learnable unitary matrix
        weight_extraction.save_quantum_weight_file(quantum_model_path)
        
        ablation = False
        if ablation:
            
            transferred_state_dict = torch.load("./transferred_weights.pth")

            conv_layer_weight_shape = transferred_state_dict['conv_layer.weight'].shape
            conv_layer_bias_shape = transferred_state_dict['conv_layer.bias'].shape

            # Initialize parameters akin to torch's default method
            fan_in = conv_layer_weight_shape[1] * conv_layer_weight_shape[2] * conv_layer_weight_shape[3]
            k_weight = 1 / torch.sqrt(torch.tensor(fan_in))

            new_weights = torch.empty(conv_layer_weight_shape)
            init.uniform_(new_weights, -k_weight.item(), k_weight.item())

            k_bias = 1 / torch.sqrt(torch.tensor(conv_layer_weight_shape[0]))

            new_bias = torch.empty(conv_layer_bias_shape)
            init.uniform_(new_bias, -k_bias.item(), k_bias.item())

            transferred_state_dict['conv_layer.weight'] = new_weights
            transferred_state_dict['conv_layer.bias'] = new_bias

            classical_model.load_state_dict(transferred_state_dict)
        else:
            classical_model.load_state_dict(torch.load("./transferred_weights.pth"))
    else:
        classical_model = models.ClassicalCNN(dataset=dataset, in_channels=1, out_channels=1, kernel_size=2, transfer=True, device=device)
        
        # Initialize both models with the same values to ensure fair comparison
        classical_model.fc1.weight.data = quantum_model.fc1.weight.data.clone()
        classical_model.fc1.bias.data = quantum_model.fc1.bias.data.clone()
        classical_model.conv_layer2.weight.data = quantum_model.conv_layer2.weight.data.clone()
        classical_model.conv_layer2.bias.data = quantum_model.conv_layer2.bias.data.clone()

    classical_optimizer = optim.Adam(classical_model.parameters(), lr=learning_rate)
    quantum_optimizer = optim.Adam(quantum_model.parameters(), lr=learning_rate)
    
    if quantum_model_path != "None":
        state_dict = torch.load(quantum_model_path)
        quantum_model.load_state_dict(state_dict)


    transfer_flag = False
    def train_and_evaluate(transfer_flag, quantum_model, quantum_optimizer, classical_model, classical_optimizer, num_epochs):
        if transfer_epoch == -1:
            epochs = num_epochs
            model = quantum_model
            optimizer = quantum_optimizer
            model_type = "Quantum"
            transfer_flag = True
        
        elif transfer_epoch == 0:
            epochs = num_epochs
            model = classical_model
            optimizer = classical_optimizer
            model_type = "Classical"
        
        else:
            if transfer_flag:
                epochs = num_epochs - transfer_epoch
                model = classical_model
                optimizer = classical_optimizer
                model_type = "Classical"
            else:
                epochs = transfer_epoch
                model = quantum_model
                optimizer = quantum_optimizer
                model_type = "Quantum"

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            train_outputs = []
            train_labels = []

            for i, (images, labels) in enumerate(train_dataloader):
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_outputs.extend(torch.sigmoid(outputs).detach().cpu().numpy())
                train_labels.extend(labels.cpu().numpy())

                print(f'Epoch [{epoch+1}/{num_epochs}], {model_type} Train Batch [{i+1}/{len(train_dataloader)}], {model_type} Train Loss: {loss.item():.4f}', end='\r')

            avg_train_loss = train_loss / len(train_dataloader)
            train_auc = roc_auc_score(train_labels, train_outputs)
            print(f'Epoch [{epoch+1}/{num_epochs}], {model_type} Average Train Loss: {avg_train_loss:.4f}, Train ROC-AUC: {train_auc:.4f}          ')

            model.eval()
            test_loss = 0.0
            test_outputs = []
            test_labels = []

            with torch.no_grad():
                for i, (images, labels) in enumerate(test_dataloader):
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    test_loss += loss.item()

                    test_outputs.extend(torch.sigmoid(outputs).detach().cpu().numpy())
                    test_labels.extend(labels.cpu().numpy())

                    print(f'Epoch [{epoch+1}/{num_epochs}], {model_type} Test Batch [{i+1}/{len(test_dataloader)}], {model_type} Test Loss: {loss.item():.4f}', end='\r')

            avg_test_loss = test_loss / len(test_dataloader)
            test_auc = roc_auc_score(test_labels, test_outputs)

            print(f'Epoch [{epoch+1}/{num_epochs}], {model_type} Average Test Loss: {avg_test_loss:.4f}, Test ROC-AUC: {test_auc:.4f}          ')

            model_save_path = os.path.join(model_save_dir, f'{model_type.lower()}_model_{current_datetime}_epoch_{epoch}.pth')
            torch.save(model.state_dict(), model_save_path)

            epoch_data = [epoch+1, model_type, avg_train_loss, train_auc, avg_test_loss, test_auc]
            
            log.training_log(log_file_path, epoch_data)



        if model_type == "Quantum" and transfer_flag == False:
            weight_extraction.save_quantum_weight_file(model_save_path)
            transfer_flag = True
            classical_model.load_state_dict(torch.load('./transferred_weights.pth'))
            train_and_evaluate(transfer_flag, quantum_model, quantum_optimizer, classical_model, classical_optimizer, num_epochs)
    
    train_and_evaluate(transfer_flag, quantum_model, quantum_optimizer, classical_model, classical_optimizer, num_epochs)
    print("Training finished")

if __name__ == "__main__":
    main()
