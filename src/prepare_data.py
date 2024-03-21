import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.utils import shuffle
import os
import binary_smiles_encoding

def prepare_dataloaders(batch_size, dataset, seed):
    
    torch.manual_seed(seed)
    dataset_file = "../datasets/"+dataset+'.npz'
    
    if not os.path.exists(dataset_file):
        print(f"Dataset file {dataset_file} not found.")
        generate = input("Generate featurized tensor? (y/n): ")
        if generate == "y":
            if not os.path.exists("../datasets/"+dataset+".csv"):
                print("Missing csv file at../datasets/"+dataset+".csv")
                print("No dataset found")
                exit()
            binary_smiles_encoding.load_csv(dataset)
            
        else:
            print("No dataset found")
            exit()
    
    data = np.load(dataset_file, allow_pickle=True)
    
    data_tensor = data["BSE"]
    label_list = data['label']

    data_tensor = torch.unsqueeze(torch.tensor(data_tensor, dtype=torch.float32),1)

    total_samples = data_tensor.shape[0]

    indices = list(range(total_samples))
    train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)
    
    np.random.seed(42)
    train_indices_shuffled = shuffle(train_indices, random_state=42)
    test_indices_shuffled = shuffle(test_indices, random_state=42)
    
    y_train = torch.tensor(label_list[train_indices_shuffled], dtype=torch.float32)
    y_test = torch.tensor(label_list[test_indices_shuffled], dtype=torch.float32)
    
    train_dataset = TensorDataset(data_tensor[train_indices_shuffled], y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    test_dataset = TensorDataset(data_tensor[test_indices_shuffled], y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_dataloader, test_dataloader
