import numpy as np
import torch

def normalize_tensors(tensor_batch, n):
    # L2 Normalize each 1D tensor in the batch
    normalized_tensors = torch.nn.functional.normalize(tensor_batch, p=2, dim=1)

    # Calculate the target length as 2^n
    target_length = 2 ** n

    # Pad each tensor to the target length with 0
    padded_tensors = torch.nn.functional.pad(normalized_tensors, (0, max(0, target_length - normalized_tensors.size(1))), "constant", 0)

    return padded_tensors


def identify_and_remove_zero_feature_vectors(tensor, device='cpu'):
    
    tensor = tensor.to(device)

    # Identify zero feature vectors and store their indices
    zero_feature_indices = torch.all(tensor == 0, dim=1).nonzero().squeeze(1)
    zero_feature_indices = zero_feature_indices.to(tensor.device)
    
    # Remove zero feature vectors from the tensor
    tensor_without_zeros = torch.index_select(tensor, 0, torch.nonzero(torch.any(tensor != 0, dim=1)).squeeze(1)).to(tensor.device)
    
    return tensor_without_zeros, zero_feature_indices
    
def insert_zero_scalars(tensor, zero_feature_indices, output_size, device='cpu'):

    tensor = tensor.to(device)
    zero_feature_indices = zero_feature_indices.to(device)  # Move zero_feature_indices to the same device as tensor
    
    # Initialize a tensor of zeros with the output size
    tensor_with_zeros = torch.zeros(output_size, dtype=tensor.dtype, device=tensor.device)
    
    # Create a mask for inserting computed values, this should be 1D
    mask = ~torch.isin(torch.arange(output_size).to(device), zero_feature_indices)
    
    # Insert the computed values where the mask is true
    tensor_with_zeros[mask] = tensor
    
    return tensor_with_zeros

def remove_duplicates(tensor, device='cpu'):
    tensor_np = tensor.detach().cpu().numpy()
    unique_tensors, indices = np.unique(tensor_np, axis=0, return_inverse=True)
    unique_tensors = torch.tensor(unique_tensors).to(device)
    return unique_tensors, indices

def repopulate_duplicates(processed_tensor, indices, original_shape):
    repopulated_tensor = processed_tensor[indices].view(original_shape)
    return repopulated_tensor
