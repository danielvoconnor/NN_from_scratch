# NN_from_scratch
This is a minimal code example that trains a feedforward neural network for MNIST classification from scratch.

You choose the number of layers, the number of nodes per layer, as well as the batch size used by the stochastic gradient method.

What's interesting is the short gradient calculation. The `forward` function is 5 lines and the `compute_grad` function is about 10 lines.

From start to finish, the code is about 150 lines. This includes preparing the MNIST data, making a plot of avg. cross-entropy vs. epoch,
and checking the classification accuracy on the validation dataset.


