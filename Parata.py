import cupy as cp
import Tanitra
import math

class Parata:

     def __init__(self,input_shape = None):
         self.params = {}
         self.input_shape = input_shape
         self.output_shape = None

     def forward(self):
         raise NotImplementedError

class PraveshParata:

    def __init__(self,input_shape):
        self.input_shape = input_shape
        self.output_shape = input_shape
        self.params = {}

    def forward(self,data):
        if data.shape == self.input_shape:
            return data
        else:
            raise ValueError("Shape of input layer and the input data should be the same.")

class GuptaParata:

    def __init__(self, n_neurons, activation, input_shape =  None):
        if isinstance(n_neurons, tuple):
            raise ValueError("You can only have a 1d NirgamParata or the number of neurons should be an integer.")
        self.output_shape = (n_neurons,)
        self.params = {}
        self.n_neurons = n_neurons
        self.input_shape = None
        self.activation = activation
        self.input_output_learned = False

    def forward(self, input):
        if not self.input_output_learned:
            self.input_shape = input.shape
            self.params = {"weights": Tanitra.Tanitra(cp.random.randn(math.prod(self.input_shape), self.n_neurons) *
                                                      1 / cp.sqrt(math.prod(self.input_shape))),
                "biases": Tanitra.Tanitra(cp.random.randn(self.n_neurons) * 0.01)}
            self.input_output_learned  = True
        input = input.flatten()
        if Tanitra.length(input) != math.prod(self.input_shape):
            raise RuntimeError("you can only input an Tanitra of the shape of input_neurons")
        output =  input @ self.params['weights'] + self.params['biases']
        if self.activation == 'relu':
            output = Tanitra.relu(output)
        elif self.activation == 'sigmoid':
            output = Tanitra.sigmoid(output)
        elif self.activation == 'linear':
            pass
        elif self.activation == 'softmax':
            output = Tanitra.softmax(output)
        else:
            raise RuntimeError("Invalid activation function was Input")
        return output

class NirgamParata:

    def __init__(self, n_neurons, activation, input_shape =  None):
        if isinstance(n_neurons, tuple):
            raise ValueError("You can only have a 1d NirgamParata or the number of neurons should be an integer.")
        self.output_shape = (n_neurons,)
        self.params = {}
        self.n_neurons = n_neurons
        self.input_shape = None
        self.activation = activation
        self.input_output_learned = False

    def forward(self, input):
        if not self.input_output_learned:
            self.input_shape = input.shape
            self.params = {"weights": Tanitra.Tanitra(cp.random.randn(math.prod(self.input_shape), self.n_neurons) *
                                                      1 / cp.sqrt(math.prod(self.input_shape))),
                "biases": Tanitra.Tanitra(cp.random.randn(self.n_neurons) * 0.01)}
            self.input_output_learned = True
        input = input.flatten()
        if Tanitra.length(input) != math.prod(self.input_shape):
            raise RuntimeError("you can only input an Tanitra of the shape of input_neurons")
        output =  input @ self.params['weights'] + self.params['biases']
        if self.activation == 'relu':
            output = Tanitra.relu(output)
        elif self.activation == 'sigmoid':
            output = Tanitra.sigmoid(output)
        elif self.activation == 'linear':
            pass
        elif self.activation == 'softmax':
            output = Tanitra.softmax(output)
        else:
            raise RuntimeError("Invalid activation function was Input")
        return output

class Samasuchaka:

    def __init__(self,normalization_type):
        self.normalization = normalization_type
        if self.normalization == 'z-score':
            self.X_mean = None
            self.y_mean = None
            self.X_std = None
            self.y_std = None
        elif self.normalization == 'min-max':
            self.X_min = None
            self.y_min = None
            self.X_max = None
            self.y_max = None
        else:
            raise ValueError("Invalid normalizer was input. Choose one out of 'z-score','min-max'.")

    def learn(self,X,y):
        if self.normalization == 'z-score':
            self.X_mean  = X.mean(axis=0)
            self.X_std = X.std(axis=0)
            self.y_mean = y.mean(axis=0)
            self.y_std = y.std(axis=0)
        if self.normalization == 'min-max':
            self.X_min = X.min(axis = 0)
            self.X_max = X.max(axis = 0)
            self.y_min = y.min(axis = 0)
            self.y_max = y.max(axis = 0)

    def forward(self,X,y):
        if self.X_mean is None or self.y_mean is None:
            raise RuntimeError("Samasuchaka must call `learn(X, y)` before `forward`")
        if self.normalization == 'z-score':
            return (X - self.X_mean) / (self.X_std+1e-8) , (y - self.y_mean) / (self.y_std+1e-8)
        if self.normalization == 'min-max':
            return (X-self.X_min)/(self.X_max-self.X_min+1e-8),(y-self.y_min)/(self.y_max-self.y_min+1e-8)


class ConvLayer2D:

    def __init__(self, stride, filters, channels, kernel_size, activation, input_shape=None, padding_constant=0,
                 padding_mode=None, padding_width=0):
        self.input_output_learned = False
        self.stride = stride
        self.filters = filters
        self.kernel_size = kernel_size
        self.input_shape = input_shape
        self.params = {}
        self.channels = channels
        self.padding = padding_mode
        self.padding_width = padding_width
        self.padding_constant = padding_constant
        self.activation = activation
        self.output = None

    def forward(self, X):
        if X.shape[0] != self.channels:
            print(X.shape)
            raise ValueError(f"Expected {self.channels} channels, but got {X.shape[0]} channels.")

        output = Tanitra.Tanitra([])
        if not self.input_output_learned:
            self.input_shape = X.shape
            self.output = (self.filters, (self.input_shape[1] - self.kernel_size) // self.stride + 1,
                           (self.input_shape[2] - self.kernel_size) // self.stride + 1)
            for i in range(self.filters):
                self.params['kernels' + str(i)] = Tanitra.Tanitra(cp.random.randn(self.channels, self.kernel_size,
                                                                                  self.kernel_size) /
                                                                  (self.input_shape[1] * self.input_shape[2]))
            self.input_output_learned = True

        for i in range(self.filters):
            feature_map = Tanitra.Tanitra(0)
            for j in range(self.channels):
                feature_map += Tanitra.convolution2d(X[j], self.params['kernels' + str(i)][j], self.stride,
                                                     padding_mode=self.padding, pad_width=self.padding_width,
                                                     constant_values=self.padding_constant)
            output = output.append(feature_map)

        if self.activation == 'relu':
            output = Tanitra.relu(output)
        elif self.activation == 'sigmoid':
            output = Tanitra.sigmoid(output)
        elif self.activation == 'linear':
            pass
        else:
            raise ValueError("Invalid activation")

        return output


class MaxPoolingLayer2D:

    def __init__(self, stride, pool_window, channels, padding_mode=None, pad_width=0, pad_constants=0,
                 input_shape=None):
        self.params = {}
        self.stride = stride
        self.pool_window = pool_window
        self.channels = channels
        self.padding = padding_mode
        self.pad_width = pad_width
        self.pad_constants = pad_constants
        self.input_shape = None
        self.output = None
        self.input_output_learned = False

    def forward(self, X):
        if X.shape[0] != self.channels:
            raise ValueError(f"Expected {self.channels} channels, but got {X.shape[0]} channels.")

        output = Tanitra.Tanitra([])
        if not self.input_output_learned:
            self.input_shape = X.shape
            self.input_output_learned = True

        for j in range(self.channels):
            output = output.append(
                Tanitra.pooling2d(X[j], self.pool_window, self.stride, padding_mode=self.padding,
                                  pad_width=self.pad_width, constant_values=self.pad_constants))

        return output

class LSTM:

    def __init__(self):
        self.long_term_memory = None
        self.short_term_memory = None
        self.params = {}
        self.params_initialized = False

    def forward(self,X):
        if not isinstance(X,Tanitra.Tanitra):
            X = Tanitra.Tanitra(X)
        if not self.params_initialized:
            self.params['forget_gate_short_memory_weights'] = Tanitra.Tanitra(cp.random.randn(len(X.data),len(X.data)))
            self.params['forget_gate_input_weights'] = Tanitra.Tanitra(cp.random.randn(len(X.data), len(X.data)))
            self.params['forget_gate_biases'] = Tanitra.Tanitra(cp.random.randn(len(X.data)))
            self.params['input_gate%_short_memory_weights'] = Tanitra.Tanitra(cp.random.randn(len(X.data), len(X.data)))
            self.params['input_gate%_input_weights'] = Tanitra.Tanitra(cp.random.randn(len(X.data), len(X.data)))
            self.params['input_gate%_biases'] = Tanitra.Tanitra(cp.random.randn(len(X.data)))
            self.params['input_gate_short_memory_weights'] = Tanitra.Tanitra(cp.random.randn(len(X.data), len(X.data)))
            self.params['input_gate_input_weights'] = Tanitra.Tanitra(cp.random.randn(len(X.data), len(X.data)))
            self.params['input_gate_biases'] = Tanitra.Tanitra(cp.random.randn(len(X.data)))
            self.params['output_gate_short_memory_weights'] = Tanitra.Tanitra(cp.random.randn(len(X.data), len(X.data)))
            self.params['output_gate_input_weights'] = Tanitra.Tanitra(cp.random.randn(len(X.data), len(X.data)))
            self.params['output_gate_biases'] = Tanitra.Tanitra(cp.random.randn(len(X.data)))
            self.params_initialized = True
        self.long_term_memory = Tanitra.Tanitra(cp.zeros_like(X.data))
        self.short_term_memory = Tanitra.Tanitra(cp.zeros_like(X.data))
        for i in range(len(X.data)):
            percentage_remember = (self.short_term_memory@self.params['forget_gate_short_memory_weights']+
                                   X[i]@self.params['forget_gate_input_weights'])+self.params['forget_gate_biases']
            percentage_remember = Tanitra.sigmoid(percentage_remember)
            self.long_term_memory *= percentage_remember
            potential_memory = (self.short_term_memory@self.params['input_gate_short_memory_weights']+
                                   X[i]@self.params['input_gate_input_weights'])+self.params['input_gate_biases']
            potential_memory = Tanitra.tanh(potential_memory)
            potential_memory_remember = (self.short_term_memory @ self.params['input_gate%_short_memory_weights'] +
                                X[i]@ self.params['input_gate%_input_weights']) + self.params['input_gate%_biases']
            potential_memory_remember = Tanitra.sigmoid(potential_memory_remember)
            self.long_term_memory += potential_memory*potential_memory_remember
            percentage_short_term_remember = (self.short_term_memory@self.params['output_gate_short_memory_weights']+
                                   X[i]@self.params['output_gate_input_weights'])+self.params['output_gate_biases']
            percentage_short_term_remember = Tanitra.sigmoid(percentage_short_term_remember)
            self.short_term_memory = percentage_short_term_remember*self.long_term_memory