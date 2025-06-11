# model_builder.py
import torch
import torch.nn as nn

class NetworkBuilder:
    def __init__(self, input_size, layer_configs, output_size, global_activation="relu", input_channels=3):
        self.input_size = input_size
        self.layer_configs = layer_configs
        self.output_size = output_size 
        self.global_activation = global_activation.lower()
        self.input_channels = input_channels

    def _get_initial_shape(self):
        if isinstance(self.input_size, tuple):
            if len(self.input_size) == 3: 
                return self.input_size
            elif len(self.input_size) == 2: 
                return (self.input_channels, self.input_size[0], self.input_size[1])
            else:
                raise ValueError("Tuple input_size must have length 2 or 3.")
        return (self.input_size,)

    def build_network(self):
        layers = []
        current_shape = self._get_initial_shape()

        for config_item in self.layer_configs:
            config = {k.lower(): v for k, v in config_item.items()}
            layer_type = config.get("type", "dense").lower()
            
            activation_name_for_layer = config.get("activation", self.global_activation).lower()

            if layer_type == "dense":
                current_shape = self._add_dense_layer(layers, config, current_shape)
            elif layer_type == "reshape": 
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
                layers.append(nn.BatchNorm2d(current_shape[0]))
            elif layer_type == "dropout":
                layers.append(nn.Dropout(config.get("probability", 0.5)))
            elif layer_type == "upsample":
                current_shape = self._add_upsample(layers, config, current_shape)
            else:
                raise ValueError(f"Layer type '{layer_type}' not supported.")

            if activation_name_for_layer != "none":
                activation_module = self._get_activation(activation_name_for_layer)
                layers.append(activation_module)
        
        return nn.Sequential(*layers)

    def _add_dense_layer(self, layers, config, current_shape):
        if len(current_shape) > 1:
             raise ValueError(f"Dense layer requires a flat input (e.g. (features,)). Got {current_shape}. Add a Flatten layer first.")
        in_features = current_shape[0]
        out_features = config.get("units", 64)
        layers.append(nn.Linear(in_features, out_features))
        return (out_features,)

    def _add_reshape_layer(self, layers, config, current_shape):
        target_shape_config = config.get("target_shape")
        if not target_shape_config:
            raise ValueError("Reshape layer must have 'target_shape'.")
        target_shape = tuple(target_shape_config) if isinstance(target_shape_config, list) else target_shape_config
        layers.append(nn.Unflatten(1, target_shape)) 
        return target_shape

    def _add_conv_layer(self, layers, config, current_shape):
        if len(current_shape) != 3:
            raise ValueError(f"Conv2D requires a 3D input (C,H,W). Got {current_shape}. Add an Unflatten/Reshape layer first.")
        in_channels = current_shape[0]
        out_channels = config.get("filters", config.get("units", 64))
        kernel_size = config.get("kernel_size", 3)
        stride = config.get("stride", 1)
        padding = config.get("padding", 0)
        use_bias = config.get("bias", False) # Often False if BatchNorm follows
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=use_bias))
        
        h_in, w_in = current_shape[1], current_shape[2]
        h_out = (h_in - kernel_size + 2 * padding) // stride + 1
        w_out = (w_in - kernel_size + 2 * padding) // stride + 1
        return (out_channels, h_out, w_out)

    def _add_transposed_conv_layer(self, layers, config, current_shape):
        if len(current_shape) != 3:
            raise ValueError(f"Conv2DTranspose requires a 3D input (C,H,W). Got {current_shape}.")
        in_channels = current_shape[0]
        out_channels = config.get("filters", config.get("units", 64))
        kernel_size = config.get("kernel_size", 4)
        stride = config.get("stride", 2)
        padding = config.get("padding", 1)
        output_padding = config.get("output_padding", 0) 
        use_bias = config.get("bias", False) # Often False if BatchNorm follows
        layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding=output_padding, bias=use_bias))

        h_in, w_in = current_shape[1], current_shape[2]
        h_out = (h_in - 1) * stride - 2 * padding + kernel_size + output_padding
        w_out = (w_in - 1) * stride - 2 * padding + kernel_size + output_padding
        return (out_channels, h_out, w_out)

    def _add_pool_layer(self, layers, config, current_shape):
        if len(current_shape) != 3:
            raise ValueError(f"Pooling layer requires 3D input (C,H,W). Got {current_shape}.")
        kernel_size = config.get("kernel_size", 2)
        stride = config.get("stride", kernel_size) 
        
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
        return (current_shape[0], h_out, w_out)

    def _add_flatten(self, layers, current_shape):
        layers.append(nn.Flatten())
        if len(current_shape) == 1: 
            return current_shape
        return (current_shape[0] * current_shape[1] * current_shape[2],)

    def _add_upsample(self, layers, config, current_shape):
        if len(current_shape) != 3:
            raise ValueError(f"Upsample layer requires 3D input (C,H,W). Got {current_shape}.")
        scale_factor = config.get("scale_factor", 2)
        mode = config.get("mode", "nearest").lower()
        if mode not in ['nearest', 'linear', 'bilinear', 'bicubic', 'trilinear']:
            raise ValueError(f"Unsupported upsample mode: {mode}")
        
        # Ensure scale_factor is int for multiplication if it came as float from config
        if isinstance(scale_factor, list): # For non-uniform scaling
            sf_h, sf_w = scale_factor[0], scale_factor[1]
        else: # Uniform scaling
            sf_h = sf_w = int(scale_factor) if isinstance(scale_factor, float) and scale_factor.is_integer() else scale_factor
        
        layers.append(nn.Upsample(scale_factor=scale_factor, mode=mode if mode != 'linear' else 'bilinear'))
        
        h_in, w_in = current_shape[1], current_shape[2]
        return (current_shape[0], int(h_in * sf_h), int(w_in * sf_w))

    def _get_activation(self, activation_key):
        activation_key = activation_key.lower()
        activations = {
            "relu": nn.ReLU(inplace=False),
            "leakyrelu": nn.LeakyReLU(0.2, inplace=False),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "softmax": nn.Softmax(dim=1)
        }
        if activation_key not in activations: # Also check for "none" here if not handled outside
            if activation_key == "none": return None # Explicitly return None for "none"
            raise ValueError(f"Activation '{activation_key}' not supported. Supported: {list(activations.keys())} or 'none'")
        return activations[activation_key]

def detect_gpu():
    if torch.cuda.is_available():
        return f"{torch.cuda.device_count()} GPU(s) - {torch.cuda.get_device_name(0)}"
    return "Aucun GPU détecté"