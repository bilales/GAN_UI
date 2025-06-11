# model_builder.py
import torch
import torch.nn as nn

class NetworkBuilder:
    def __init__(self, input_size, layer_configs, output_size, global_activation="relu", input_channels=3):
        """
        input_size: an int for a latent vector (generator) or a tuple (for image dimensions).
                   If a tuple of length 3 is provided, it's assumed to be (channels, height, width).
                   Otherwise, if a tuple of length 2 is provided, input_channels is used as the channel count.
        layer_configs: list of layer configuration dictionaries.
        output_size: (NOW LARGELY INFORMATIONAL) for the generator, the flattened image size; for the discriminator, number of output channels.
        global_activation: default activation function if not specified in layer_config.
        input_channels: number of channels for convolutional inputs (if input_size is a 2-tuple).
        """
        self.input_size = input_size
        self.layer_configs = layer_configs
        self.output_size = output_size # This is now mostly for reference, not for adding a layer
        self.global_activation = global_activation.lower()
        self.input_channels = input_channels

    def _get_initial_shape(self):
        if isinstance(self.input_size, tuple):
            if len(self.input_size) == 3: # (C, H, W)
                return self.input_size
            elif len(self.input_size) == 2: # (H, W)
                return (self.input_channels, self.input_size[0], self.input_size[1])
            else:
                raise ValueError("Tuple input_size must have length 2 or 3.")
        return (self.input_size,) # (latent_dim,) or (flattened_features,)

    def build_network(self):
        layers = []
        current_shape = self._get_initial_shape()

        for config_item in self.layer_configs:
            config = {k.lower(): v for k, v in config_item.items()}
            layer_type = config.get("type", "dense").lower()
            
            activation_name_for_layer = config.get("activation", self.global_activation).lower()

            if layer_type == "dense":
                current_shape = self._add_dense_layer(layers, config, current_shape)
            elif layer_type == "reshape": # This is nn.Unflatten in PyTorch
                current_shape = self._add_reshape_layer(layers, config, current_shape)
            elif layer_type in ["conv2d", "convolution"]:
                current_shape = self._add_conv_layer(layers, config, current_shape)
            elif layer_type in ["conv2dtranspose", "transposed_conv"]:
                current_shape = self._add_transposed_conv_layer(layers, config, current_shape)
            elif layer_type in ["maxpool", "avgpool"]:
                current_shape = self._add_pool_layer(layers, config, current_shape)
            elif layer_type == "flatten":
                current_shape = self._add_flatten(layers, current_shape)
            elif layer_type == "batchnorm1d":
                if len(current_shape) != 1:
                    raise ValueError(f"BatchNorm1d expects a 1D input. Got shape: {current_shape}")
                layers.append(nn.BatchNorm1d(current_shape[0]))
            elif layer_type == "batchnorm2d":
                if len(current_shape) != 3:
                     raise ValueError(f"BatchNorm2d expects a 3D input (C,H,W). Got shape: {current_shape}")
                layers.append(nn.BatchNorm2d(current_shape[0])) # Assumes current_shape[0] is channels
            elif layer_type == "dropout":
                layers.append(nn.Dropout(config.get("probability", 0.5)))
            elif layer_type == "upsample":
                current_shape = self._add_upsample(layers, config, current_shape)
            # "unflatten" was a custom name, your config uses "Reshape" for nn.Unflatten
            # So, the "reshape" case above handles it.
            else:
                raise ValueError(f"Layer type '{layer_type}' not supported.")

            if activation_name_for_layer != "none":
                activation_module = self._get_activation(activation_name_for_layer)
                layers.append(activation_module)
        
        # --- MODIFIED: Removed unconditional output layer addition ---
        # The layer_configs from config.json should define the ENTIRE network.
        
        return nn.Sequential(*layers)

    def _add_dense_layer(self, layers, config, current_shape):
        if len(current_shape) > 1:
             raise ValueError(f"Dense layer requires a flat input (e.g. (features,)). Got {current_shape}. Add a Flatten layer first.")
        in_features = current_shape[0]
        out_features = config.get("units", 64)
        layers.append(nn.Linear(in_features, out_features))
        return (out_features,)

    def _add_reshape_layer(self, layers, config, current_shape): # PyTorch nn.Unflatten
        target_shape_config = config.get("target_shape")
        if not target_shape_config:
            raise ValueError("Reshape layer must have 'target_shape'.")
        
        # Ensure target_shape is a tuple
        target_shape = tuple(target_shape_config) if isinstance(target_shape_config, list) else target_shape_config

        # nn.Unflatten(dim, unflattened_size)
        # Assumes input to this layer will be (batch_size, features_to_unflatten)
        # current_shape is (features_to_unflatten,)
        # We unflatten along dim 1 (the feature dimension)
        layers.append(nn.Unflatten(1, target_shape)) 
        return target_shape # (C, H, W)

    def _add_conv_layer(self, layers, config, current_shape):
        if len(current_shape) != 3: # Expects (C, H, W)
            raise ValueError(f"Conv2D requires a 3D input (C,H,W). Got {current_shape}. Add an Unflatten/Reshape layer first.")
        in_channels = current_shape[0]
        out_channels = config.get("filters", config.get("units", 64))
        kernel_size = config.get("kernel_size", 3)
        stride = config.get("stride", 1)
        padding = config.get("padding", 0)
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)) # bias=False often with BatchNorm
        
        # Calculate output H, W
        h_in, w_in = current_shape[1], current_shape[2]
        h_out = (h_in - kernel_size + 2 * padding) // stride + 1
        w_out = (w_in - kernel_size + 2 * padding) // stride + 1
        return (out_channels, h_out, w_out)

    def _add_transposed_conv_layer(self, layers, config, current_shape):
        if len(current_shape) != 3: # Expects (C, H, W)
            raise ValueError(f"Conv2DTranspose requires a 3D input (C,H,W). Got {current_shape}.")
        in_channels = current_shape[0]
        out_channels = config.get("filters", config.get("units", 64))
        kernel_size = config.get("kernel_size", 4)
        stride = config.get("stride", 2)
        padding = config.get("padding", 1)
        # output_padding is sometimes needed to reach exact target dimensions
        output_padding = config.get("output_padding", 0) 
        layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding=output_padding, bias=False))
        # Note: Your original had BatchNorm2d here. It's often better to specify BatchNorm2d explicitly in config if needed.
        # If you want it always after ConvTranspose, keep it. Otherwise, remove and add to config.
        # layers.append(nn.BatchNorm2d(out_channels)) # Example: if you always want it

        h_in, w_in = current_shape[1], current_shape[2]
        h_out = (h_in - 1) * stride - 2 * padding + kernel_size + output_padding
        w_out = (w_in - 1) * stride - 2 * padding + kernel_size + output_padding
        return (out_channels, h_out, w_out)

    def _add_pool_layer(self, layers, config, current_shape):
        if len(current_shape) != 3:
            raise ValueError(f"Pooling layer requires 3D input (C,H,W). Got {current_shape}.")
        kernel_size = config.get("kernel_size", 2)
        stride = config.get("stride", kernel_size) # Stride often defaults to kernel_size for pooling
        
        layer_type = config.get("type", "maxpool").lower()
        if layer_type == "maxpool":
            layers.append(nn.MaxPool2d(kernel_size, stride))
        elif layer_type == "avgpool":
            layers.append(nn.AvgPool2d(kernel_size, stride))
        else:
            raise ValueError(f"Unknown pooling type: {layer_type}")

        h_in, w_in = current_shape[1], current_shape[2]
        h_out = (h_in - kernel_size) // stride + 1
        w_out = (w_in - kernel_size) // stride + 1
        return (current_shape[0], h_out, w_out) # Channels remain the same

    def _add_flatten(self, layers, current_shape):
        layers.append(nn.Flatten())
        if len(current_shape) == 1: # Already flat
            return current_shape
        # Calculate flattened size: C * H * W
        return (current_shape[0] * current_shape[1] * current_shape[2],)

    def _add_upsample(self, layers, config, current_shape):
        if len(current_shape) != 3:
            raise ValueError(f"Upsample layer requires 3D input (C,H,W). Got {current_shape}.")
        scale_factor = config.get("scale_factor", 2)
        mode = config.get("mode", "nearest").lower()
        if mode not in ['nearest', 'linear', 'bilinear', 'bicubic', 'trilinear']:
            raise ValueError(f"Unsupported upsample mode: {mode}")
        layers.append(nn.Upsample(scale_factor=scale_factor, mode=mode if mode != 'linear' else 'bilinear')) # 'linear' is for 1D, use 'bilinear' for 2D
        
        h_in, w_in = current_shape[1], current_shape[2]
        # Ensure scale_factor is int for multiplication if it came as float from config
        sf = int(scale_factor) if isinstance(scale_factor, float) and scale_factor.is_integer() else scale_factor
        return (current_shape[0], h_in * sf, w_in * sf)


    def _get_activation(self, activation_key):
        activation_key = activation_key.lower()
        activations = {
            "relu": nn.ReLU(inplace=False),
            "leakyrelu": nn.LeakyReLU(0.2, inplace=False),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "softmax": nn.Softmax(dim=1) # dim=1 assumes batch dimension is 0
        }
        if activation_key not in activations:
            raise ValueError(f"Activation '{activation_key}' not supported. Supported: {list(activations.keys())}")
        return activations[activation_key]

def detect_gpu():
    if torch.cuda.is_available():
        return f"{torch.cuda.device_count()} GPU(s) - {torch.cuda.get_device_name(0)}"
    return "Aucun GPU détecté"