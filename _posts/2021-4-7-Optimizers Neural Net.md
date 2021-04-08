---
layout: post
title: Overview of Optimizer Architecture in Neural Nets
---

A gentle guide to Optimizer Architecture in Neural Nets, in plain English
----

## Quick Primer on Optimization

At a very high level, a neural network runs numerous iterations during training:
- a forward pass to calculate their outputs based on the current parameters and the input data
- a loss function to calculate a 'cost' for the gap between the current outputs and the desired target outputs
- a backward pass to calculate the gradients of the loss relative to the parameters
- an optimization step that uses the gradients to update the parameters in such a way as to reduce the loss for the next iteration

There are many types of optimizers that perform the final step above, with a goal of progressively reducing the loss in each iteration, till they find the minimum loss for the network.

Instead, it might be easier to imagine each of those millions of parameters independently. We can picture a 2D curve with only 1 parameter. Think of this as keeping all other parameters constant and varying only this single parameter and plotting the loss and gradients for it. Then imagine that there are millions of such separate curves.

Also, just to be clear, although the trajectory of the visualization shows you going 'down the slope' as you traverse the 3D curve, the optimizer itself operates only on the horizontal 2D parameter plane, and only indirectly moves the loss along the vertical dimension.

The optimizer updates the parameter value from one point to another on the horizontal axis. In turn, in the next iteration, the corresponding loss gets reduced along the vertical axis.

So, what we've just seen is that, depending on your starting point within the Loss Landscape, which you select randomly, gradient descent might encounter landscape features that are very hard to traverse.

## Overview of how Optimization is done in a Neural Network

Picture of Model Parameters. Parameter Groups showing a CNN section and a Linear Classifier section which are put into different Parameter Groups. Training Hyperparameters like Learning Rate, Momentume Rate etc. Then show where the Optimizer comes in - what inputs do they take and what they are updating. Then show where Schedulers come in - what inputs do they take and what are they updating.

## Quick Summary of popular Optimizers and Schedulers

## Conclusion


