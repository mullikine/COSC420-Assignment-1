#!/bin/bash
export TTY

# change into the lab directory before running perceptron.py

cd lab

# All of these parameters are optional and they override parameters taken from
# param.txt

# -I number of input units
# -H number of hidden units
# -O number of output units

# -R disables randomization

# -W SHOW WEIGHTS in standard error
# -A SHOW ACTIVATIONS in standard error

# -a set the activation function. Possible values: sigmoid, tanh, rectifier, leaky_relu, softmax, analytic
# -E set the error function. Possible values: sum_of_squares, negative_log_likelihood

# -r learning rate. Possible values: (0.0 to 1.0)

# -e specifies the maximum number of epochs

# -l regularization factor. Possible values: (0.0 to 1.0)
# -m momentum. Possible values: (0.0 to 1.0)
# -d learning rate tdecay. Possible values: (0.0 to 1.0)

# -t training data. A relative path

# -M MODE. Passible values: learn

../perceptron.py -W -A \
    -R  \
    -e10000  \ 
    -r0.5 \
    -l0.0 \
    -m0.5 \
    -d0.001 \
    -a sigmoid \
    -t "../datasets/Iris" \
    -o results/my.time_vs_error.log