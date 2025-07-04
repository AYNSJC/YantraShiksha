import cupy as cp
from cupyx.scipy.signal import convolve2d

# Core autodiff variable class
class Tanitra:

    def __init__(self, data, track_gradient=True):
        self.data = cp.array(data)                    # Tensor data stored as CuPy array for GPU acceleration
        self.track_gradient = track_gradient          # Whether to track gradient for this node
        self.parents = []                             # Parents in computation graph for backpropagation
        self.shape = self.data.shape                  # Cached shape
        self.grad = None                              # Gradient w.r.t. this variable

    # Operator overloads:

    # Elementwise addition
    def __add__(self, other):
        if not isinstance(other, Tanitra):
            other = Tanitra(other)
        new_tanitra = Tanitra(self.data + other.data)
        if self.track_gradient:
            # Grad of a + b w.r.t a is 1, w.r.t b is 1
            new_tanitra.parents.append((self, lambda g: g * cp.ones_like(other.data)))
            new_tanitra.parents.append((other, lambda g: g * cp.ones_like(self.data)))
        return new_tanitra

    # Elementwise subtraction
    def __sub__(self, other):
        if not isinstance(other, Tanitra):
            other = Tanitra(other)
        new_tanitra = Tanitra(self.data - other.data)
        if self.track_gradient:
            new_tanitra.parents.append((self, lambda g: g * cp.ones_like(other.data)))
            new_tanitra.parents.append((other, lambda g: -g * cp.ones_like(self.data)))
        return new_tanitra

    # Elementwise multiplication
    def __mul__(self, other):
        if not isinstance(other, Tanitra):
            other = Tanitra(other)
        new_tanitra = Tanitra(self.data * other.data)
        if self.track_gradient:
            new_tanitra.parents.append((self, lambda g: g * other.data))
            new_tanitra.parents.append((other, lambda g: g * self.data))
        return new_tanitra

    # Elementwise division
    def __truediv__(self, other):
        if not isinstance(other, Tanitra):
            other = Tanitra(other)
        new_tanitra = Tanitra(self.data / other.data)
        if self.track_gradient:
            new_tanitra.parents.append((self, lambda g: g / other.data))
            new_tanitra.parents.append((other, lambda g: -g * self.data / (other.data ** 2)))
        return new_tanitra

    # Matrix multiplication
    def __matmul__(self, other):
        if not isinstance(other, Tanitra):
            other = Tanitra(other)
        new_tanitra = Tanitra(self.data @ other.data)
        if self.track_gradient:
            a_shape = self.shape
            b_shape = other.shape
            new_tanitra.parents.append((self, lambda g: (g.reshape(-1, b_shape[-1]) @ other.data.T).reshape(a_shape)))
            new_tanitra.parents.append((other, lambda g: (self.data.T @ g.reshape(-1, b_shape[-1])).reshape(b_shape)))
        return new_tanitra

    # Indexing / slicing
    def __getitem__(self, index):
        new_tanitra = Tanitra(self.data[index])
        if self.track_gradient:
            def gradn(grad):
                gradient = cp.zeros_like(self.data)
                gradient[index] += grad
                return gradient
            new_tanitra.parents.append((self, gradn))
        return new_tanitra

    # Appending tensors along an axis
    def add(self, obj, axis=0):
        if not isinstance(obj, Tanitra):
            obj = Tanitra(obj)
        self.data = cp.append(self.data, obj.data, axis=axis)

    # Flatten the tensor
    def flatten(self):
        new_tanitra = Tanitra(self.data.flatten())
        if self.track_gradient:
            new_tanitra.parents.append((self, lambda g: g.reshape(self.shape)))
        return new_tanitra

    # Dot product
    def dot(self, a):
        if not isinstance(a, Tanitra):
            a = Tanitra(a)
        result = cp.dot(self.data, a.data)
        new_tanitra = Tanitra(result)
        new_tanitra.parents.append((a, lambda g: g * self.data))
        new_tanitra.parents.append((self, lambda g: g * a))
        return new_tanitra

    # Append for nested lists
    def append(self, other):
        if not isinstance(other, Tanitra):
            other = Tanitra(other)
        self_list = self.data.tolist()
        other_list = other.data.tolist()
        length = len(self_list)
        self_list.append(other_list)
        self.data = cp.array(self_list)
        new_tanitra = Tanitra(self.data)
        if self.track_gradient or other.track_gradient:
            new_tanitra.parents.append((self, lambda g: g[:length]))
            new_tanitra.parents.append((other, lambda g: g[-1]))
        return new_tanitra

    # Backward pass: compute gradients
    def backward(self, grad=None):
        if grad is None:
            grad = cp.ones_like(self.data)
        if self.grad is None:
            self.grad = grad
        else:
            self.grad += grad
        for parent, gradient_function in self.parents:
            parent.backward(gradient_function(grad))

    # Reset all gradients
    def grad_0(self):
        for parent, _ in self.parents:
            parent.grad = 0
            parent.grad_0()
        self.parents = []

    # Transpose
    def T(self):
        x = Tanitra(self.data.T)
        if self.track_gradient:
            x.parents.append((self, lambda g: g.T))
        return x

# Activation functions

def sigmoid(data):
    if not isinstance(data, Tanitra):
        data = Tanitra(data)
    result = 1 / (1 + cp.exp(-data.data))
    new_tanitra = Tanitra(result)
    if data.track_gradient:
        new_tanitra.parents.append((data, lambda g: g * result * (1 - result)))
    return new_tanitra

def relu(data):
    if not isinstance(data, Tanitra):
        data = Tanitra(data)
    result = cp.maximum(0, data.data)
    new_tanitra = Tanitra(result)
    if data.track_gradient:
        new_tanitra.parents.append((data, lambda g: g * (data.data > 0).astype(float)))
    return new_tanitra

def tanh(x):
    if not isinstance(x, Tanitra):
        x = Tanitra(x)
    result = cp.tanh(x.data)
    new_tanitra = Tanitra(result)
    if x.track_gradient:
        new_tanitra.parents.append((x, lambda g: g * (1 - result ** 2)))
    return new_tanitra

def sin(x):
    if not isinstance(x, Tanitra):
        x = Tanitra(x)
    result = cp.sin(x.data)
    new_tanitra = Tanitra(result)
    if x.track_gradient:
        new_tanitra.parents.append((x, lambda g: g * cp.cos(x.data)))
    return new_tanitra

def cos(x):
    if not isinstance(x, Tanitra):
        x = Tanitra(x)
    result = cp.cos(x.data)
    new_tanitra = Tanitra(result)
    if x.track_gradient:
        new_tanitra.parents.append((x, lambda g: -g * cp.sin(x.data)))
    return new_tanitra

# Element-wise operations

def square(data):
    if not isinstance(data, Tanitra):
        data = Tanitra(data)
    result = cp.square(data.data)
    new_tanitra = Tanitra(result)
    if data.track_gradient:
        new_tanitra.parents.append((data, lambda g: g * 2 * data.data))
    return new_tanitra

def log(x):
    if not isinstance(x, Tanitra):
        x = Tanitra(x)
    x.data = cp.clip(x.data, 1e-9, None)
    result = cp.log(x.data)
    new_tanitra = Tanitra(result)
    if x.track_gradient:
        new_tanitra.parents.append((x, lambda g: g / x.data))
    return new_tanitra

def mean(data, axis=None):
    return Tanitra(data.data.mean(axis=axis))

def length(data):
    if not isinstance(data, Tanitra):
        raise TypeError("length function is only for Tanitra class objects.")
    return len(data.data)

def to_cons(data):
    return data.data

# Softmax function (for classification)
def softmax(a, axis=0):
    if not isinstance(a, Tanitra):
        a = Tanitra(a)
    maxed = a.data - cp.max(a.data, axis=axis, keepdims=True)
    exp = cp.exp(maxed)
    summation = cp.sum(exp, axis=axis, keepdims=True)
    result = exp / summation
    new_tanitra = Tanitra(result)
    if a.track_gradient:
        new_tanitra.parents.append((a, lambda g: g * result * (1 - result)))
    return new_tanitra

# Convolution operation with stride and padding
def convolution2d(a, b, stride, padding_mode=None, pad_width=0, constant_values=0):
    if padding_mode is not None:
        a_padded = cp.pad(a.data, pad_width=pad_width, mode=padding_mode, constant_values=constant_values)
    else:
        a_padded = a.data
    b_flipped = cp.flip(b.data)
    output = convolve2d(b_flipped, a_padded, mode='valid')[::stride, ::stride]
    new_tanitra = Tanitra(output)

    if a.track_gradient or b.track_gradient:
        # Gradient w.r.t input a
        def gradient_function_a(grad):
            if stride < 1:
                raise RuntimeError("Invalid Stride")
            elif stride == 1:
                grad_a = convolve2d(grad, b_flipped, mode='full')
            else:
                H_out = ((a_padded.shape[0] - b_flipped.shape[0]) // stride) + 1
                W_out = ((a_padded.shape[1] - b_flipped.shape[1]) // stride) + 1
                H_grad = H_out + (H_out - 1) * (stride - 1)
                W_grad = W_out + (W_out - 1) * (stride - 1)
                unsampled = cp.zeros((H_grad, W_grad))
                unsampled[::stride, ::stride] = grad
                grad_a = convolve2d(unsampled, b_flipped, mode='full')
            if padding_mode is not None:
                if isinstance(pad_width, tuple):
                    grad_a = grad_a[pad_width[0][0]:-pad_width[0][1], pad_width[1][0]:-pad_width[1][1]]
                else:
                    grad_a = grad_a[pad_width:-pad_width, pad_width:-pad_width]
            return grad_a

        # Gradient w.r.t kernel b
        def gradient_function_b(grad):
            unsampled = cp.zeros((grad.shape[0] + (grad.shape[0] - 1) * (stride - 1),
                                  grad.shape[1] + (grad.shape[1] - 1) * (stride - 1)))
            unsampled[::stride, ::stride] = grad
            return convolve2d(a_padded, unsampled, mode='valid')

        new_tanitra.parents.append((a, gradient_function_a))
        new_tanitra.parents.append((b, gradient_function_b))
    return new_tanitra

# Max pooling with stride
def pooling2d(a, pool_size, stride, padding_mode=None, pad_width=None, constant_values=0):
    if padding_mode is not None:
        a_padded = cp.pad(a.data, pad_width=pad_width, mode=padding_mode, constant_values=constant_values)
    else:
        a_padded = a.data

    m, n = a_padded.shape
    h = (m - pool_size) // stride + 1
    w = (n - pool_size) // stride + 1
    new_tanitra_data = cp.zeros((h, w))
    indices_list = []

    for i in range(h):
        for j in range(w):
            region = a.data[i:i + pool_size, j:j + pool_size]
            new_tanitra_data[i, j] = cp.max(region)
            index = cp.argmax(region)
            indices_list.append((index // pool_size + i * stride, index % pool_size + j * stride))

    new_tanitra = Tanitra(new_tanitra_data)

    if a.track_gradient:
        def gradient_function(grad):
            unsampled = cp.zeros_like(a_padded)
            row_indices, col_indices = zip(*indices_list)
            unsampled[row_indices, col_indices] = grad.reshape(len(row_indices))
            if padding_mode is not None:
                if isinstance(pad_width, tuple):
                    unsampled = unsampled[pad_width[0][0]:-pad_width[0][1], pad_width[1][0]:-pad_width[1][1]]
                else:
                    unsampled = unsampled[pad_width:-pad_width, pad_width:-pad_width]
            return unsampled
        new_tanitra.parents.append((a, gradient_function))
    return new_tanitra
