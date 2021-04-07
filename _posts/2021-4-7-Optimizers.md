---
layout: post
title: Overview of Optimizers
---

A gentle guide to Optimizers, in plain English
----

Overview of the most useful optimizers that you will commonly used - SGD, Momentum, RMSProp, Adam

Additional articles - go into more depth on each one - Exponential Moving Avg + Momentum, Adam + AdamW etc

For instance, both Pytorch and Keras have a plethora of built-in optimizers like Adagrad, Adam, etc. What are there so many different ones? How do we choose which one to use? If you read the docs for each one, they describe a formulae for the weight update. But what does each formula mean? Let's first try to get some overall context so we can then get some intuition about how each one fits in, before diving into the formulae.

# Article 1 Overview of how Optimization is done in a Neural Network
This could be a complete article by itself.

Picture of Model Parameters. Parameter Groups showing a CNN section and a Linear Classifier section which are put into different Parameter Groups. Training Hyperparameters like Learning Rate, Momentume Rate etc. Then show where the Optimizer comes in - what inputs do they take and what they are updating. Then show where Schedulers come in - what inputs do they take and what are they updating.

## Overview of Optimization with Gradient Descent
- Explain what is Loss Landscape
- Start with a typical picture of 3D loss landscape with a convex curve, Weights w1/w2 on the horizontal plane and Loss on the vertical plane. Show a trajectory going down the curve.
- Explain why this is an idealised scenario, not realistic. Say that this is a nice visualization to get the initial concept intuition when you are starting out, but this is not realistic for a few reasons.
- One reason is that this is an ideal convex curve. In reality the curve is very bumpy
- Second reason is that this trajectory shows you traversing the 3D curve. whereas actually the optimizer is only traversing the horizontal flat 2D plane of the parameters. You aren't actually going "down the slope". The Loss reduces in the next iteration, but the optimizer only operates on the weights.
- Third reason is that there are not just 2 parameters. There are billions and it is impossible to visualize that or even imagine that in your head.
- So instead draw a 2D curve with only 1 parameter. On that show what gradient means by showing delta y/delta x which corresponds to the slope. Then show that Optimizer only updates weights on the X axis. The Loss function on the Y axis changes indirectly as a consequence.
- Then say that what I find more helpful to visualise is the curve for a single parameter independently. This means that keep all other params constant, and only change this one parameter and plot the Loss and gradients for that. Then imagine that there are a billion such separate curves.

## Challenges with Optimization using Gradient Descent
- You might have many minima. So you might get stuck in a local minima and not reach the global. Gradient Descent finds it very difficult to jump out of a local minima.
- Basically you could have unusual shapes inside the curve that become hard to traverse.
- You could have saddle points. KD - what exactly is the challenge with saddle points? Is it that the gradient is steep in one direction, but the actual direction you want to go is a gentle slope.
- You could have pathological curvature where there is a narrow ravine. And what you want to do is not go down the side of the ravine to the bottom of the valley, but actually traverse along the valley. because there is a minima at the end of the valley eg. sort of like a river valley that ends in a lake or the sea.
- So depending on your starting point within the Loss Landscape, which you select randomly, you might encounter landscape features which become hard to traverse.

## First Improvement to Gradient Descent - Stochastic Gradient Descent
Here rather than take the full dataset, we operate on a randomly selected mini batch at a time. That randomness helps us explore the loss landscape. Since there are different data samples in each mini batch, the Loss value will vary. And so also the gradients will vary. In effect you are actually varying the Loss Landscape with each mini batch. So even though in one mini batch you might end up in a topography where you could get stuck, the next mini batch you might end up in a different place so that lets you keep moving. So that helps you not to get stuck in a particular section of the landscape, especially in the early stages of training.

## Second Improvement to Gradient Descent - Momentum
One of the tricky parts of Gradient Descent is dealing with steep slopes. It is very easy to take a large step when you actually want to go slowly and cautiously. Because the gradient is large there, you could make a large update and jump out of the valley altogether.

Now ideally you want to be a bit intelligent about this based on where in the landscape you are. If the slope is very steep you want to slow down. If the slope is very flat you might want to speed up and so on. So you want to vary the magnitude of the update dynamically so you can respond to changes in the landscape around you.

With Gradient Descent you make an update to the weights at each step, based on the gradient and the learning rate. In other words, to modify the size of the update, there are two things you can do:
- One is to adjust the gradient
- Second is to adjust the learning rate

Momentum is a way to do the first above ie. adjust the gradient. Right now we look only at the current gradient, and ignore all the past gradients. This means that if there is a sudden change in the landscape due to an anomaly in the curve, your trajectory may get thrown off course. That is sort of like saying there is a sudden ditch or cliff and you react to just that without using the knowledge of the surrounding landscape that you had seen up until that point.

So the idea is that you let the past gradients guide your overall direction so that you stay on course. The question becomes how far in the past do you go? and does every gradient from the past count equally? From intuition it would make sense that things from the recent past should count more than things from the distant past. So that if the change in landscape is not an anomaly, but a genuine structural change, then you do need to react to it bit by bit and change your course gradually.

This helps you tackle problems like the pathological curvature. Basically this means that one weight parameter the gradient is very high and for another it is very low. So with SGD, you might actually bounce around from one side of the valley to another. But with momentum, you can dampen the oscillations. Because the large gradients from one step for the first parameter cancels out the large gradient updates from the next step in the opposite direction. On the other hand, the small updates for the first step for the second parameter reinforces the small updates for the second step because they are in the same direction.

There are different techniques like Plain Momentum, Nestorov Gradient which do this using different formulae.

## Third Improvement to Gradient Descent - Modify Learning Rate (based on the landscape)
Here we modify the learning rate component using different formulae.
For instance, if they square the gradients, then the cancelling out effect from opposite directions, that we talked about in Momentum, will get negated. So if we are in a steep slope, the gradients are large and the square of the gradients are large and always positive, so they accumulate fast. To dampen this, the formula makes a bigger adjustment to reduce the learning rate. For shallower slopes, the accumulation is small and so the formula makes a smaller adjustment to the learning rate.

There are a few different optimizer techniques which do this. eg. Adagrad, Adadelta, RMS Prop.

And then there are some which build on top of the Momentum improvements, so they are a combination, modifying both the gradient as well as the learning rate. eg. Adam, and its many variants, LAMB.

# Article 2 Fourth Improvement to Gradient Descent - Modify Learning Rate (based on your training progress)
This topic is actually not under Optimizers at all. This is handled by Schedulers. They vary the learning rate and other optimization hyperparameters based on a formula that depends on the epoch of the training.

# Article 3 Details of Optimizers - Momentum with Exponential Moving Avg, RMSProp and Adam
Include some of my Pytorch code from optimiser_lib to show the Python Step code for calculating each optimiser.

RMSProp adjusts learning rate separately for each parameter. And so the formula applies to the projection ie. the component of the gradient that lies along the direction/axis of the parameter we're updating.

## Conclusion


