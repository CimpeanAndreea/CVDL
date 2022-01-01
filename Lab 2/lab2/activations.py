import numpy as np

def softmax(x, t=1):
    """"
    Applies the softmax temperature on the input x, using the temperature t

    t change the final probabilities ->
    as t goes to 0, the model will be "more confident" about it's prediction
    """
    # TODO your code here

    # softmax does not represent the inputs distribution well for inputs too large or too small
    x_stabilized = x - np.max(x)
    return np.exp(x_stabilized / t) / np.sum(np.exp(x_stabilized / t))

    # end TODO your code here