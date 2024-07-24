# Building an Autograd Library from Scratch in C for Simple Neural Networks

## Content
<!-- toc -->

## Introduction
Autograd, short for automatic differentiation, is a fundamental component in machine learning frameworks, enabling the automatic computation of gradients that are used for training neural networks. This article will walk you through my journey of writing an autograd library from scratch in pure C.

The source code is hosted in this [repository](https://github.com/smdaa/teeny-autograd-c)

## A Neural Network: A Brief Overview
At its core, a neural network consists of neurons organized in layers. Each neuron receives input from the previous layer, processes it using a weighted sum, applies an activation function, and passes the output to the next layer.

<p align="center">
  <img width=600 src="neuron.png">
</p>

Mathematically, we can express the output of a single neuron as:
$$
y = f(\sum_{i=1}^{n} w_i x_i + b)
$$

where \\(x_i\\)​ are the inputs, \\(w_i\\)​​ are the weights, \\(b\\) is the bias, and \\(f\\) is the activation function.

A layer is simply a collection of neurons, and neural networks typically consist of three types of layers: input, hidden, and output layers.

<p align="center">
  <img width=300 src="neuron-network.png">
</p>

Neural networks learn by adjusting the weights \\(w_i\\) and biases \\(b\\) of each neuron to minimize the error in their predictions. This is done via [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent), where the network computes the gradient of the error with respect to each weight and bias, and then updates them in the direction that reduces the error.

## Methods of Derivative Calculation: Numerical, Analytical, and Automatic Differentiation
Computing gradients is necessary for a neural network to learn, There are 3 fundamental ways to calculate derivatives:

* **Symbolic differentiation**: It involves finding the exact derivative of a function using algebraic rules. If the function has a known mathematical expression, we can compute its derivative symbolically. For example for \\(f(x) = x^2\\) the derivative is \\(f'(x) = 2x\\). This method can be computationally expensive and lead to unwieldy expressions.

* **Numerical differentiation**: This method uses finite differences to approximate the derivative of a function, a common practice in fields like aerospace engineering and fluid dynamics. \\(f'(x) = \frac{f(x+h) - f(x)}{h}\\). However this method is not suited for neural networks since  there are a large number of derivatives to be performed in a neural network, and with automatic differentiation we can do better in term of performance.

* **Automatic differentiation**: 