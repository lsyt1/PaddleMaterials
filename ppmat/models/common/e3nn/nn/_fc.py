import paddle
import paddle.nn as nn

class FullyConnectedNet(nn.Layer):
    def __init__(self, layers, activation=None, var_in=1.0, var_out=1.0):
        super().__init__()
        self.layers = nn.LayerList()
        for h1, h2 in zip(layers[:-1], layers[1:]):
            self.layers.append(nn.Linear(h1, h2))
        self.activation = activation or nn.ReLU()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.layers) - 1:
                x = self.activation(x)
        return x

class _Layer(nn.Layer):
    def __init__(self, h1, h2, act, var_in, var_out):
        super().__init__()
        tensor = paddle.randn((h1, h2))
        self.weight = paddle.create_parameter(
            shape=tensor.shape, 
            dtype=tensor.dtype, 
            default_initializer=paddle.nn.initializer.Assign(tensor)
        )
