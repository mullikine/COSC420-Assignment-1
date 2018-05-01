import numpy as np

np.seterr(all = 'ignore')

def wrap(pre, post):
    def decorate(func):
        def call(*args, **kwargs):
            pre(func, *args, **kwargs)
            result = func(*args, **kwargs)
            post(func, *args, **kwargs)
            return result
        return call
    return decorate

# using softmax as output layer is recommended for classification where outputs are mutually exclusive
def softmax(w):
    e = np.exp(w - np.amax(w))
    dist = e / np.sum(e)
    return dist

# using tanh over logistic sigmoid for the hidden layer is recommended
def tanh(x):
    return np.tanh(x)


# def mysum(x):
#     """
#     sum of array elements.
#     """
#     return np.sum(x)


def random_normal(inputs, hiddens):
    # use scheme from Efficient Backprop by LeCun 1998 to initialize weights for hidden layer
    input_range = 1.0 / inputs ** (1/2)

    return np.random.normal(loc = 0,
                            scale = input_range,
                            size = (inputs, hiddens))

def random_uniform(outputs, hiddens):
    return np.random.uniform(size = (hiddens, outputs)) / np.sqrt(hiddens)
