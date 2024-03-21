import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import utils
import quantum_layers
   
class QuantumConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, filter_size=2, strides=1, bias=True, device=torch.device('cpu'), vqc_layers=3):
        super(QuantumConvolution, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_size = filter_size
        self.strides = strides
        self.input_data_per_circuit = in_channels*(filter_size**2)
        self.device = device

        self.QuantumLayer = nn.ModuleList()
        for _ in range(self.out_channels):
            layer = quantum_layers.AmplitudeQuantumLayer(input_data_per_circuit=self.input_data_per_circuit, vqc_layers=vqc_layers).to(device)
            self.QuantumLayer.append(layer)
        
        self.bias = bias
        if bias == True:
            # Mimic pytorch conv2d bias intialization 
            k = 1.0 / (in_channels * math.prod((filter_size, filter_size)))
            self.QuantumLayerBias = nn.Parameter(torch.empty(out_channels).uniform_(-math.sqrt(k), math.sqrt(k))).to(device)


    def forward(self, x):
        
        if len(x.shape) == 3:
            x = torch.unsqueeze(input=x, axis=1)

        batch_size, _, input_height, input_width = x.shape
    
        # Calculate output dimensions
        output_height = (input_height - self.filter_size) // self.strides + 1
        output_width = (input_width - self.filter_size) // self.strides + 1

        # This operation extracts sliding windows and arranges them into a new dimension
        unfolded = x.unfold(2, self.filter_size, self.strides).unfold(3, self.filter_size, self.strides)
        
        # unfolded shape: [batch_size, channels, output_height, output_width, filter_size, filter_size]
        output_array = unfolded.permute(0, 2, 3, 1, 4, 5).flatten(start_dim=2).reshape(batch_size * output_height * output_width, -1).to(self.device)

        # All-zeros vectors do not constitute a valid quantum state, however we know this inner product should produce 0
        # Save the indices so that the 0-valued inner-product may be reinserted later
        x_features, x_feature_indices = utils.identify_and_remove_zero_feature_vectors(output_array)
        
        # Reduce computational overhead by only passing unique states to the quantum device
        # Save the indices so the results may be copied and inserted later
        x_features_unique, x_features_unique_indices = utils.remove_duplicates(x_features)   
        
        stacked = []

        for i in range(self.out_channels):
            
            x_features_unique = utils.normalize_tensors(x_features_unique, math.ceil(math.log2(x_features_unique.shape[-1])))

            # Run the quantum circuit and insert the inner product results for identical states
            QL_output = utils.repopulate_duplicates(self.QuantumLayer[i](x_features_unique) , x_features_unique_indices, x_features.shape[0])

            # Insert the zero-valued dot product where the vectors were all zeros
            QL_output = utils.insert_zero_scalars(QL_output, x_feature_indices, output_array.size(0),device=self.device).view(batch_size,output_height,output_width)              

            if self.bias:
                 QL_output = QL_output + self.QuantumLayerBias[i]          

            stacked.append(QL_output)

        return torch.stack(stacked,dim=1).view(batch_size,self.out_channels,output_height,output_width)
    
class FilterNormalizationConv2DParallel(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(FilterNormalizationConv2DParallel, self).__init__()
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None

    def forward(self, x):
        # Calculate output dimensions
        height, width = x.shape[2], x.shape[3]
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # Unfold the input tensor to extract sliding windows
        unfolded = F.unfold(x, kernel_size=self.kernel_size, dilation=1, padding=self.padding, stride=self.stride)
        
        # Normalize the unfolded tensor
        unfolded_norm = F.normalize(unfolded, p=2, dim=1)
        
        # Reshape weight for matrix multiplication
        weight_reshaped = self.weight.view(self.out_channels, -1)
        
        # Perform matrix multiplication between the normalized unfolded tensor and the weights
        # Adding a new dimension to unfolded_norm to support broadcasting when multiplying
        output = torch.matmul(weight_reshaped, unfolded_norm).view(-1, self.out_channels, out_height, out_width)
        
        if self.bias is not None:
            # Adding bias to each output channel
            output += self.bias.view(1, self.out_channels, 1, 1)
        
        return output

