---
title: Dimension tracking for Convolutional Layers in PyTorch and Tensorflow
author: Turzo Bose
layout: post
icon: fa-lightbulb
icon-style: regular
---
Keeping track of the dimensions is a very cumbersome process if not properly understood, specially while designing the deep neural network model architectures. The work adds on when having to shift between different machine learning frameworks, ideally PyTorch and Tensorflow. It is crazy how although both these frameworks are ideally doing the same thing, but the syntax makes it increasingly difficult to allow for fluid transitions between them. Hence, this post is written to make the dimension tracking process simpler, for both PyTorch and Tensorflow implementations.

## PyTorch

PyTorch has the built-in **Conv2d** class that allows for the 2D convolutional operation over a input plane (e.g. think about an input image of shape 28 X 28 X 3 - height X width X channels).

Lets say we want to convolve the image with a kernel/filter of size 3 (e.g. 3 X 3) and want an output with 5 channels

The **Conv2d** class takes in the following arguments:

`nn.Conv2d(in_channels = 3, out_channels = 5, kernel_size = 3, stride = 1, padding = 0)     #can create non-square kernels with kernel_size = (2,3)`

Here, the input channel comes from the input image, the output channel is defined by us, and the stride = 1 and padding = 0 is by default, but can be changed as per need. We can also define a non-square kernel of 2X3 by defining kernel_size = (2,3).

We might be wondering what does the number of *out_channels* actually mean. It actually corresponds to the **number of kernels/filters** we want the layer to have. Lets understand this with some code:

`import torch
input_batch = torch.rand(16, 3, 100, 100) # N = Batch Size, C = Input Channels(RGB),
                                    # H = Height , W = Width
conv = torch.nn.Conv2d(
		in_channels=3, # RGB channels
		out_channels=7, # Number of kernels/filters this layer has
		kernel_size=5, # Size of kernels, i. e. of size 5x5
		stride=1,
		padding=0)
print(conv.weight.size()) # 7 x 3 x 5 x 5 (7 kernels of size 5x5 having depth of 3 )
print(conv(input_batch).size()) # 16 x 7 x 100 x 100 => Batch Size = 16, Channels = 7
                                                        Height = 100, Width = 100`

Now, let us calculate how the dimensions are calculated. The formula for the dimensionality is:

$$Dimensions = \lfloor\dfrac{N+2p-f}{s} + 1\rfloor$$

Hence, with a stride of 1 and padding of 2, the output size computes to $$\dfrac{100+2*2-5}{1} + 1 = 100$$. So, the output size is 100 x 100.

## Tensorflow

Tensorflow supports Keras, which has the built-in **Conv2D** layer class that allows for the 2D convolutional operation over a input plane (*notice the subtle different in uppercase D differing from lowercase d in PyTorch*)

Lets try to do the same convolution operation as above with Tensorflow.

The **Conv2D** layer class takes in the following arguments:

`tf.keras.layers.Conv2D(filters = 3, kernel_size = 3, stride = 1, padding = "valid")(input_tensor)     #can create non-square kernels with kernel_size = (2,3)`

Here, filters is the same as *out_channels* as PyTorch, and refers to the number of filters we are assigning to the layer. The rules for non-square kernel size and stride is the same as PyTorch, but for padding, one of "valid":no padding or "same":no loss in dimensions (case-insensitive) has to be added.

One key difference in the implementations is the **data_format**. Tensorflow uses channels_last as default, but PyTorch uses channels_first format. So, we need to be very careful for this. Lets understand this with some code:

`import tensorflow as tf
input_batch = torch.rand(16, 3, 100, 100) # N = Batch Size, C = Input Channels(RGB),
                                    # H = Height , W = Width
conv = torch.nn.Conv2d(
		in_channels=3, # RGB channels
		out_channels=7, # Number of kernels/filters this layer has
		kernel_size=5, # Size of kernels, i. e. of size 5x5
		stride=1,
		padding=0)
print(conv.weight.size()) # 7 x 3 x 5 x 5 (7 kernels of size 5x5 having depth of 3 )
print(conv(input_batch).size()) # 16 x 7 x 100 x 100 => Batch Size = 16, Channels = 7
                                                        Height = 100, Width = 100`


When using this layer as the first layer in a model, provide the keyword argument input_shape (tuple of integers, does not include the sample axis), e.g. input_shape=(128, 128, 3) for 128x128 RGB pictures in data_format="channels_last".
