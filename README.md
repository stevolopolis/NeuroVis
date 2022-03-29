# NeuroVis -- A neuroscience way of visualizing CNNs.

This repository contains a number of CNN visualization techniques specialized for neuroscience research using PyTorch.

Besides the conventional methods such as Activation Maximization (AM) and its variants, this repository also contains methods borrowed from the neuroscience field:
- Feature maps (i.e. neuron activation maps) of each layer.
- Kernel ranking for various rankings such as "orientation".
- Group AM on an entire layer or batches of kernels.

## Tables of Content
* [Background](#background)
* [Activation Maximization](#activation-maximization)
* [Group AM](#group-am)
* [Feature Map Visualization](#feature-map-visualization)
* [Kernel Ranking](#kernel-ranking)

## Background

Recent progress in deep learning has taken a lot of intuitions from the human brain. From the simple multi-layer perceptron to the variations of CNNs, these architecture share substantial similarity with how the brain processes sensory information (e.g. V1, V2, etc.). With the growth in computational power, these neural networks have been showing performances that exceed the human mind in specific downstream tasks. 

However, till this day, the inner-workings of these powerful neural networks are still in question. While there are methods for visualizing weights of neural networks such as activation maximization and gradient-based saliency maps, the process of understanding individual weights in the network are still unintuitive to scientists. 

NeuroVis tries to tackle this problem from a different perspective.

## Activation Maximization

## Group AM

## Feature Map visualization

## Kernel Ranking