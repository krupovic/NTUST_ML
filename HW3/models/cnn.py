import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class CNN(nn.Module):
    """
    Flexible CNN that adapts to different input shapes and user-specified depth.
    
    For MNIST: Expects (batch, 1, 28, 28) - uses Conv2d
    For Adult/tabular: Expects (batch, features) - reshapes and uses Conv1d
    """
    
    def __init__(self,
                 input_shape: tuple,  # (channels, height, width) for 2D or (features,) for 1D
                 num_classes: int,
                 num_conv_layers: int = 3,
                 base_channels: int = 32,
                 channel_multiplier: float = 2.0,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 pool_size: int = 2,
                 fc_sizes: Optional[List[int]] = None,
                 dropout: float = 0.5,
                 activation: nn.Module = nn.ReLU):
        """
        Args:
            input_shape: Input dimensions
            num_classes: Number of output classes
            num_conv_layers: Number of convolutional layers
            base_channels: Starting number of channels
            channel_multiplier: Multiply channels by this factor each layer
            kernel_size: Convolution kernel size
            stride: Convolution stride
            padding: Convolution padding
            pool_size: Max pooling size
            fc_sizes: Fully connected layer sizes (if None, uses [512, 256])
            dropout: Dropout probability
            activation: Activation function class
        """
        super().__init__()
        
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.activation = activation()
        self.dropout = nn.Dropout(dropout)
        
        # Determine if we're working with 1D (tabular) or 2D (image) data
        if len(input_shape) == 1:
            # 1D tabular data - use Conv1d
            self.data_type = '1d'
            self.input_channels = 1
            self.feature_dim = input_shape[0]
            conv_layer = nn.Conv1d
            pool_layer = nn.MaxPool1d
            adaptive_pool = nn.AdaptiveAvgPool1d(1)
        elif len(input_shape) == 2:
            # 2D image data (assuming grayscale, can extend for RGB)
            self.data_type = '2d'
            self.input_channels = 1
            self.height, self.width = input_shape
            conv_layer = nn.Conv2d
            pool_layer = nn.MaxPool2d
            adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        elif len(input_shape) == 3:
            # 3D image data (channels, height, width)
            self.data_type = '2d'
            self.input_channels, self.height, self.width = input_shape
            conv_layer = nn.Conv2d
            pool_layer = nn.MaxPool2d
            adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            raise ValueError(f"Unsupported input shape: {input_shape}")
        
        # Build convolutional layers
        self.conv_layers = nn.ModuleList()
        in_channels = self.input_channels
        
        for i in range(num_conv_layers):
            out_channels = int(base_channels * (channel_multiplier ** i))
            
            if self.data_type == '1d':
                conv = conv_layer(in_channels, out_channels, kernel_size, stride, padding)
                pool = pool_layer(pool_size)
            else:
                conv = conv_layer(in_channels, out_channels, kernel_size, stride, padding)
                # Only add pooling for first few layers to avoid over-reduction
                # For deep networks, skip pooling in later layers
                if i < min(3, num_conv_layers - 1):  # Pool at most 3 times, and not on last layer
                    pool = pool_layer(pool_size)
                else:
                    pool = nn.Identity()  # No pooling
            
            self.conv_layers.append(nn.Sequential(
                conv,
                nn.BatchNorm1d(out_channels) if self.data_type == '1d' else nn.BatchNorm2d(out_channels),
                self.activation,
                pool
            ))
            in_channels = out_channels
        
        # Adaptive pooling to get fixed-size feature vector
        self.adaptive_pool = adaptive_pool
        self.final_conv_channels = in_channels
        
        # Fully connected layers
        if fc_sizes is None:
            fc_sizes = [512, 256]
        
        fc_layers = []
        in_features = self.final_conv_channels
        
        for fc_size in fc_sizes:
            fc_layers.extend([
                nn.Linear(in_features, fc_size),
                self.activation,
                self.dropout
            ])
            in_features = fc_size
        
        # Output layer
        if num_classes == 2:
            # Binary classification - single output with sigmoid
            fc_layers.append(nn.Linear(in_features, 1))
            self.output_type = 'binary'
        else:
            # Multiclass - softmax output
            fc_layers.append(nn.Linear(in_features, num_classes))
            self.output_type = 'multiclass'
        
        self.fc_layers = nn.Sequential(*fc_layers)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Reshape input based on data type
        if self.data_type == '1d':
            if x.dim() == 2:  # (batch, features) -> (batch, 1, features)
                x = x.unsqueeze(1)
        elif self.data_type == '2d':
            if x.dim() == 2:  # Flatten MNIST: (batch, 784) -> (batch, 1, 28, 28)
                if hasattr(self, 'height') and hasattr(self, 'width'):
                    x = x.view(batch_size, self.input_channels, self.height, self.width)
                else:
                    # Assume square image
                    side = int(x.size(1) ** 0.5)
                    x = x.view(batch_size, self.input_channels, side, side)
        
        # Pass through convolutional layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        # Adaptive pooling and flatten
        x = self.adaptive_pool(x)
        x = x.view(batch_size, -1)
        
        # Pass through fully connected layers
        x = self.fc_layers(x)
        
        return x
    
    def get_num_parameters(self):
        """Return total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_cnn_for_dataset(dataset: str, num_conv_layers: int = 3, base_channels: int = 32, 
                          fc_sizes: Optional[List[int]] = None, **kwargs) -> CNN:
    """
    Factory function to create CNN appropriate for specific datasets
    """
    if dataset.lower() == 'mnist':
        # MNIST: 28x28 grayscale images, 10 classes
        input_shape = (28, 28)
        num_classes = 10
    elif dataset.lower() == 'adult':
        # Adult: will be determined at runtime from data loader, but typically ~100 features after preprocessing
        # We'll use a placeholder and update it when we see the actual data
        input_shape = (100,)  # Placeholder - will be updated
        num_classes = 2
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    
    if fc_sizes is None:
        fc_sizes = [512, 256] if dataset.lower() == 'mnist' else [128, 64]
    
    return CNN(
        input_shape=input_shape,
        num_classes=num_classes,
        num_conv_layers=num_conv_layers,
        base_channels=base_channels,
        fc_sizes=fc_sizes,
        **kwargs
    )


if __name__ == "__main__":
    # Quick test
    print("Testing CNN architectures...")
    
    # Test MNIST-style CNN
    cnn_mnist = create_cnn_for_dataset('mnist', num_conv_layers=3, base_channels=32)
    print(f"MNIST CNN: {cnn_mnist.get_num_parameters():,} parameters")
    
    # Test with dummy MNIST data
    dummy_mnist = torch.randn(4, 784)  # Batch of 4, flattened 28x28 images
    output_mnist = cnn_mnist(dummy_mnist)
    print(f"MNIST output shape: {output_mnist.shape}")
    
    # Test Adult-style CNN
    cnn_adult = create_cnn_for_dataset('adult', num_conv_layers=2, base_channels=16)
    print(f"Adult CNN: {cnn_adult.get_num_parameters():,} parameters")
    
    # Test with dummy Adult data
    dummy_adult = torch.randn(4, 100)  # Batch of 4, 100 features
    output_adult = cnn_adult(dummy_adult)
    print(f"Adult output shape: {output_adult.shape}")
