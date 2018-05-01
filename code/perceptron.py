#!/usr/bin/env python2
# -*- coding: utf-8 -*-
""" Create the percepton and repeat epochs to classify some data. The final result is printed when population error is less than the user specified error criterion.
Example: ../perceptron.py -R -e100 -H10 -r0.1 -l0.0 -m0.5 -d0 -a sigmoid -t "../datasets/Iris" -o results/regularization-0-0-100i-iris-sigmoid.time_vs_error.log
"""

# todo:
# The final result is printed out when the population error is less than user specified error criterion.

import sys
import os
import time
import random
from sympy import *

#  from utils import *
from my_numpy import *
# import numpy as np
from numpy import zeros, ones, dot, shape, reshape, log, loadtxt

from os import linesep

def print_error(message):
    """Print to stderror"""
    sys.stderr.write(str(message) + linesep)
0
# debug

if __name__ == '__main__':
    # Default parameters
    # ------------------
    # Default

    show_weights = False
    show_activations = False
    randomize = True # disable when debugging
    log_path = "log.txt"
    mode = "learn"                # learn/test/show
    g_inputs = 2
    g_hiddens = 2
    g_outputs = 1
    g_learning_rate = 0.01
    g_regularization_factor = 0.0 # No regularization
    g_epochs = 10
    g_momentum = 0.5
    g_decay_rate = 0.0001
    g_criterion = 0.02

    g_activation_function = 'sigmoid'
    g_error_function = 'sum_of_squares'

    pop_error = 0.0

    #  sigmoid function is better suited for two-class logistic regression and
    #  softmax is used for the multiclass logistic regression (e.g. Maximum
    #  Entropy Classifier).
    # TODO: softmax

    # Globals
    # -------
    g_input_activations = None
    g_hidden_activations = None
    g_output_activations = None


    import getopt

    try:
        opts, args = getopt.getopt(sys.argv[1:], "I:H:O:l:e:E:r:m:d:a:t:o:M:DRWA",
                                   ['inputs=',
                                    'hiddens=',
                                    'outputs=',
                                    'learning_rate=',
                                    'epochs=',
                                    'momentum=',
                                    'decay_rate=',
                                    'regularization=',
                                    'activation_fuction=',
                                    'training_data=',
                                    'show_weights=',
                                    'show_activations=',
                                    'output_file'])

    except getopt.GetoptError as err:
        print_error(str(err))
        sys.exit(1)


    # -M learn
    # -M test
    # -M show

    for option, argument in opts:
        if option in ("-I", "--inputs"):
            g_inputs = int(argument)
        elif option in ("-H", "--hiddens"):
            g_hiddens = int(argument)
        elif option in ("-O", "--outputs"):
            g_outputs = int(argument)
        elif option in ("-R", "--norandomize"):
            randomize = False
        elif option in ("-W", "--show-weights"):
            show_weights = True
        elif option in ("-A", "--show-activations"):
            show_activations = True
        elif option in ("-a", "--activation-function"):
            g_activation_function = str(argument)
        elif option in ("-E", "--error-function"):
            g_error_function = str(argument)
        elif option in ("-r", "--learning-rate"):
            g_learning_rate = float(argument)
        elif option in ("-e", "--epochs"):
            g_epochs = int(argument)
        elif option in ("-l", "--regularization-factor"):
            g_regularization_factor = float(argument)
        elif option in ("-m", "--momentum"):
            g_momentum = float(argument)
        elif option in ("-d", "--decay"):
            g_decay_rate = float(argument)
        elif option in ("-t", "--training-data"):
            training_data_path = argument
            if os.path.exists(training_data_path + "/param.txt"): # Do this here so I can override it
                param_data = loadtxt(training_data_path + "/param.txt", dtype=float)
                g_inputs = int(param_data[0])
                g_hiddens = int(param_data[1])
                g_outputs = int(param_data[2])
                g_learning_rate = float(param_data[3])
                g_momentum = float(param_data[4])
                g_criterion = float(param_data[5])
        elif option in ("-o", "--outputfile"):
            log_path = argument
        elif option in ("-M", "--mode"):
            mode = argument


    #  sys.exit(0)


    #  input_data what the thing we are trying to classify looks like

    #  can't use. np.asmatrix
    #  from ptpython.repl import embed; embed(globals(), locals())
    #  input_data = np.asmatrix(loadtxt(training_data_path + "/in.txt", dtype=float))
    #  input_data = np.atleast_2d(loadtxt(training_data_path + "/in.txt", dtype=float, delimiter=" "))
    input_data = loadtxt(training_data_path + "/in.txt", dtype=float, ndmin=2)

    n_examples = input_data.shape[0]

    g_inputs = input_data.shape[1]

    #  output_data what the thing we are trying to classify should be classified as
    #  one hot encoded
    #  can't use. np.asmatrix
    #  output_data = np.asmatrix(loadtxt(training_data_path + "/out.txt", dtype=float))

    # .reshape((-1, 1)) This would make output 1-dimensional
    #output_data = np.atleast_2d(loadtxt(training_data_path + "/out.txt", dtype=float, delimiter=" "))

    lines = open(training_data_path + "/out.txt", "r").readlines()

    output_data = loadtxt(training_data_path + "/out.txt", dtype=float, ndmin=2)

    #  from ptpython.repl import embed; embed(globals(), locals())
    g_outputs = output_data.shape[1]

    #  print_error(input_data)
    #  print_error(output_data)
    #  print_error(g_inputs)
    #  print_error(g_outputs)
    #  sys.exit(0)

    # globals are truly evil in python

    # set up array of 1s for activations
    g_input_activations = ones(g_inputs + 1) # +1 for bias
    g_hidden_activations = ones(g_hiddens)
    g_output_activations = ones(g_outputs)

    # Zero momentum changes
    last_input_changes = zeros((g_inputs + 1, g_hiddens)) # +1 for bias. Layer 1 and 2
    last_output_changes = zeros((g_hiddens, g_outputs))   # Layer 2 and 3

    # Randomize weights
    g_input_weights = random_normal(g_inputs + 1, g_hiddens) # +1 for bias
    g_output_weights = random_uniform(g_outputs, g_hiddens)

    # Disable randomization by setting the seed
    if not randomize:
        np.random.seed(0)

    # else:
    #     g_input_weights = ones((g_inputs, g_hiddens))
    #     g_output_weights = ones((g_outputs, g_hiddens))

    print_error("Running with parameters:")
    print_error("------------------------")
    print_error(str(g_epochs) + " max epochs")
    print_error(str(g_criterion) + " g_criterion")
    print_error(str(g_inputs) + " input units and 1 bias unit")
    print_error(str(g_hiddens) + " hidden units")
    print_error(str(g_outputs) + " output units")
    print_error("learning rate = " + str(g_learning_rate))
    print_error("regularization factor = " + str(g_regularization_factor))
    print_error("momentum = " + str(g_momentum))
    print_error("decay rate = " + str(g_decay_rate))

    start = time.time()

    # combine examples (input_data) with targets (output_data) to form teaching_patterns
    teaching_patterns = []

    # create tuples of (input pattern, target) pairs
    for i in range(input_data.shape[0]):
        teaching_patterns.append(list((list(input_data[i,:]), list(output_data[i]))))


    def array_to_column(a):
        """Turn array into column"""
        return reshape(a, (a.shape[0],1))

                                                                     # print_error(teaching_patterns)

    def activation(x, f=g_activation_function):     # See: https://theclevermachine.wordpress.com/tag/tanh-function/
        """The activation function"""

        if f == "sigmoid":                          # Logistic sigmoid. When x is negative, this gives < 0.5. When positive, it gives > 0.5
            return 1 / (1 + np.exp(-x))
        elif f == "tanh":                           # The hyperbolic [tangent] activation function. stronger gradients, avoid getting stuck, avoid bias in gradient
            return tanh(x)
        elif f == "rectifier":
                                                    # return np.maximum(x, 0)
            return x * (x > 0)
        elif f == "leaky_relu":                     # TODO: rectified linear unit (ReLU)
            return np.maximum(x, 0)
        elif f == "softmax":                        # TODO: often used in the final layer of a neural network-based classifier
            e = np.exp(x - np.amax(x))              # np.amax: Return the maximum of an array or maximum along an axis. np.array([5,4,2]) - 1 = array([4, 3, 1])
            return e / np.sum(e)
        elif f == "analytic":                       # TODO: softplus. a smooth approximation to relu. derivative of sigmoid
            return np.log(1 + np.exp(x))
        else:
            return x                                # The identity activation function

    def activation_derivative(y, f=g_activation_function):
        """The activation function"""

        if g_activation_function == "sigmoid":      # slope of the sigmoid function in order to determine how much the weights need to change
            return y * (1.0 - y)
        elif g_activation_function == "tanh":
            return 1. - y*y
        elif g_activation_function == "rectifier":
            return 1. * (x > 0)
        elif g_activation_function == "leaky_relu": # TODO: leaky relu
            alpha=0.01                              # this could be a parameter
            dx = np.ones_like(y)
            dx[x < 0] = alpha
            return dx
        elif g_activation_function == "softmax":
            return y                                # TODO
        else:
            return y

    def activation_delta(outputs, targets, f=g_activation_function):
        """Calculate delta. How for from target and in what direction."""

        # delta/theta tells you the direction to change the weights.

        # Depending
        # on the activation function used previously, we calculate the delta
        # differently

        # The core idea behind back-prop, backward differentiation # https://www.quora.com/How-do-you-understand-the-Delta-in-Back-Propagation-Algorithm

        if f == "sigmoid": # Logistic sigmoid
            return activation_derivative(outputs, f) * -(targets - outputs)
        elif f == "tanh":
            return activation_derivative(outputs, f) * dot(outputs, targets) # not sure if correct
        elif f == "rectifier":
            return x
        elif f == "softmax":
            return -(targets - outputs)
        elif f == "analytic":
            return x
        else:
            return x

    def error(targets, outputs, f=g_error_function):
        """The error function"""

        if f == "sum_of_squares":
            return sum(0.5 * (targets - outputs)**2)
        elif f == "negative_log_likelihood":
            return -sum(targets * np.log(outputs)) # In practice, the softmax function is used in tandem with the negative log-likelihood. Becomes unhappy at smaller values
        else:
            return 0

    global pop_error

    # Train the neural network in batch mode
    for i in range(1, g_epochs + 1): # range = 1 to (g_epochs = 100) inclusive
        epoch_error = 0.0

        if randomize:
            random.shuffle(teaching_patterns)

        for pat in teaching_patterns:

            # DONE Set all the activations for the hidden layer
            # DONE Set all the activations for the output layer
            # Find and set the error for the output layer
            # Find and set the error for the hidden layer
            # Weight and Bias adjustment for the output layer
            # Weight and Bias adjustment for the hidden layer
            # Calculate the population error
                                                                         # See: file://references.org "feedforward"

            g_input_activations[0:g_inputs] = np.array(pat[0])                       # shape: (4,1). The thing we want to classify. len(g_input_activations) == g_inputs+1. Don't overwrite the bias (the last item in list).

            # a = wx + b: Because the last item in g_input_activations is 1 we get bias when we do the dot product
            g_hidden_activations = activation(dot(g_input_weights.T, g_input_activations))    # dot_product.shape=(10,1). g_input_weights.shape=(4.10). multiply by the weights and run activation function.
            g_output_activations = activation(dot(g_output_weights.T, g_hidden_activations))  # We desire for these to equal the training data



                                                                         # See: file://references.org "backprop"
            targets = np.array(pat[1])                                   # shape: (3,1). used for back prop

            # Must calculate output error to calculate hidden error.
            #
            # These values are small
            output_deltas = activation_delta(g_output_activations, targets)                                    # shape: (3,1)
            #  activation_derivative(g_hidden_activations).shape
            #  dot(g_output_weights, output_deltas).shape
            #  from ptpython.repl import embed; embed(globals(), locals())
            hidden_deltas = activation_derivative(g_hidden_activations) * dot(g_output_weights, output_deltas) # shape: (2,1)

            def change_weights(deltas, activations, weights, previous_change):
                """Delta Rule
                Training is done by subtracting from each weight a small fraction of its corresponding derivative
                This function is idempotent.
                Also implements regularization and momentum.
                """

                #  from ptpython.repl import embed; embed(globals(), locals())
                change = np.multiply(deltas, array_to_column(activations)) # shape: (2,3)
                regularization = g_regularization_factor * weights

                #  the delta increment for a weight = the learning rate constant times the gradient associated with the weight.
                weights -= (g_learning_rate * change + regularization) + (previous_change * g_momentum) # shape: (2,3). The weights change here

                return weights, change

            #  from ptpython.repl import embed; embed(globals(), locals())

            g_input_weights, last_input_changes = change_weights(hidden_deltas, g_input_activations, g_input_weights, last_input_changes)
            g_output_weights, last_output_changes = change_weights(output_deltas, g_hidden_activations, g_output_weights, last_output_changes)

            epoch_error += error(targets, g_output_activations)

        #  from ptpython.repl import embed; embed(globals(), locals())

        t = str(time.time() - start)
        #print_error(t)

        with open(log_path, 'a') as logfile:
            logfile.write(t + "," + str(epoch_error) + '\n')
            logfile.close()

        # for each output node, calculate the error (difference) between the
        # output node and the pattern, then find the popuwlation error[<0;92;35M
        pop_error = epoch_error / (float(g_outputs) * n_examples)

        if show_weights or show_activations or pop_error < g_criterion or i == g_epochs:
            print_error("Epoch %d, Population error %-.2f" % (i, pop_error))

        #  from ptpython.repl import embed; embed(globals(), locals())
        if show_weights or i == g_epochs:
            items=["Input Weights", g_input_weights,
                   "Output Weights", g_output_weights]
            for p in items: print_error(p)

        if show_activations or i == g_epochs:
            items=["Input activations", g_input_activations[0:g_inputs],
                   "Hidden activations", g_hidden_activations]
            for p in items: print_error(p)

        if show_weights or show_activations or pop_error < g_criterion:
            print_error("\n")

        if pop_error < g_criterion:
            break

        try:
            # decay
            if g_learning_rate> 0 and g_decay_rate > 0:
                g_learning_rate = g_learning_rate * (g_learning_rate / (g_learning_rate + (g_learning_rate * g_decay_rate)))


            if i % 100 == 0:
                # Population error must print
                print_error("Epoch %d, Population error %-.2f" % (i, pop_error))

        except Exception as e:
            print_error(e)

            from ptpython.repl import embed; embed(globals(), locals())

    if pop_error < g_criterion:
        print_error(str(pop_error) + " < " + str(g_criterion) + "; Error criterion was reached")
    else:
        print_error(str(pop_error) + " >= " + str(g_criterion) + "; Error criterion was not reached")

    sys.exit(0)