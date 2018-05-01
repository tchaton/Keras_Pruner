# Keras_Pruner

## Efficient Keras implementation of  [\[1611.06440 Pruning Convolutional Neural Networks for Resource Efficient Inference\]](https://arxiv.org/abs/1611.06440) ##

# Only the code of the pruner is present.

# Process :
  - Create pipeline using Augmentor on my repo
  - Feed the trained neural network to the Transformer (wrap layers into a custom Wrapper)
  - Feed this model to NVIDIA_Pruner, set shapes (#could be implemented inside this class)
  - WAIT FOR PRUNING

