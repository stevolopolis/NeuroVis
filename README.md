# NeuroVis -- A neuroscience way of visualizing CNNs.

This repository contains a number of CNN visualization techniques specialized for neuroscience research using PyTorch.

Besides the conventional methods such as Activation Maximization (AM) and its variants, this repository also contains methods borrowed from the neuroscience field:
- Feature maps (i.e. neuron activation maps) of each layer.
- Kernel ranking for various rankings such as "orientation".
- Group AM on an entire layer or batches of kernels.

## Tables of Content
* [Background](#background)
* [Files to interact with](#files)
* [Activation Maximization](#activation-maximization)
* [Group AM](#group-am)
* [Feature Map Visualization](#feature-map-visualization)
* [Kernel Ranking](#kernel-ranking)

## Files to interact with

```main_am.py```

```neuro_activation.py```

```am_guided_backprop.py```

```integrated_grad.py```


## Background

Recent progress in deep learning has taken a lot of intuitions from the human brain. From the simple multi-layer perceptron to the variations of CNNs, these architecture share substantial similarity with how the brain processes sensory information (e.g. V1, V2, etc.). With the growth in computational power, these neural networks have been showing performances that exceed the human mind in specific downstream tasks. 

However, till this day, the inner-workings of these powerful neural networks are still in question. While there are methods for visualizing weights of neural networks such as activation maximization and gradient-based saliency maps, the process of understanding individual weights in the network are still unintuitive to scientists. 

NeuroVis tries to tackle this problem from a different perspective.

## Activation Maximization
![AM-example](/media/conv1_0.png)
*AM visualization matrix of the 1st layer and 0th kernel.*
The format of each AM visualization matrix is as follows:
| | | | |
|---|---|---|---|
| Initial image (rgb) | AM result (rgb) | - | target feature map | 
| Initial image (depth) | AM result (depth) | - | Resulting feature map

## Neuron Activation Maps
![Neuron Activation Map of Conv3 Layer](/media/0_1ec297183c8aa37a36c7d12bccd8bbd__conv3_0.png)
*Activation map of the 3rd Conv layer 0th kernel. (Black: low activation; White: is high activation. )*


## Saliency Maps