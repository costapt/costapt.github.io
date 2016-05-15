---
layout:     post
title:      "Batch Normalization"
subtitle:   "How to implement the Batch Norm layer"
date:       2016-04-25 12:00:00
author:     "Pedro Costa"
header-img: "img/batch-norm-cover.jpg"
---

I am doing the CS231N stanford's course which is wonderful bla bla bla

# Batch Normalization

One preprocessing technique widely used across every Machine Learning algorithm
is to normalize the input features to have zero mean and unit variance. In
practice, this technique tends to make algorithms that are optimized with
gradient descent converge faster to the solution.

One way we can look at deep neural networks is as stacks of different models
(layers), where the output of one model is the input of the next.
And so the question is: can't we normalize the output of each layer?
That is what [Ioffe et al, 2015](http://arxiv.org/abs/1502.03167) proposed with
the Batch Normalization layer.

In order to be able to introduce the normalization in the neural network's
training pipeline, it should be fully differentiable (or at least almost
everywhere differentiable like the ReLU function).
The good news is that it is fully differentiable, but let's see a little
example.

Assume the output of an hidden layer $$X$$ is an $$(N,D)$$ matrix, where
$$N$$ is the number of examples present in the batch and $$D$$ is the number of
hidden units. We start by normalizing $$X$$:

$$ \hat{X} = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}\ ,$$

where $$\mu_B$$ is the mean of the batch and $$\sigma_B^2$$ is the variance.
As $$\mu_B$$ and $$\sigma_B^2$$ are differentiable, we can see that $$ \hat{X}$$
is also differentiable, and so we are good to go.

Then, the authors did something really clever. They realized that by
normalizing the output of a layer they could limit its representational power
and, therefore, they wanted to make sure that the Batch Norm layer could fall
back to the identity function.

$$ y_i = \gamma \hat{x}_i + \beta $$

Note that when $$\gamma = \sqrt{\sigma_B^2 + \epsilon}$$ and $$\beta = \mu_B$$
the Batch Norm simply outputs the previous layer's activations.
The network has the ability to ignore the Batch Norm layer if it is the optimal
thing to do.
As you might have guessed by now, $$\gamma$$ and $$\beta$$ are learnable
parameters that are initialized with $$\gamma = 1$$ and $$\beta = 0$$.

The authors claim that this layer has a regularizing effect and that we no
longer need to use dropout. For more information, I recommend that you read the
[paper](http://arxiv.org/abs/1502.03167). It is very well written and everything
is explained better than I could. Impressive results.

The authors claim that Batch Norm has several nice properties such as:

- **Reduce internal covariate shift**:
[Ioffe et al, 2015](http://arxiv.org/abs/1502.03167) define internal covariate
shift as "the change in the distribution of the network activations due to the
change in network parameters during training". With Batch Norm, the next layer
can expect the same input distribution at each iteration.
- **Regularization effect**: It is claimed that with Batch Norm we could
disregard or reduce the strength of Dropout. The value given to one training
example depends on the other examples in the batch, since the mean and variance
are computed on the batch level and, therefore, it is not deterministic. This
property can increase the generalization of the network.
- **Reduce probability of vanishing or exploding gradients**: As the Batch Norm
layer is placed prior to the non-linearity, it prevents the training to get
stuck on saturated areas of non-linearities, solving the issue of vanishing
gradients. Also, it makes the network more robust to the scale of the
parameters.

In practice, this allows us to increase the learning rate and train the network
with a lot less iterations and, as this was not enough, get better
results.

For more information, read the [paper](http://arxiv.org/abs/1502.03167). It is
very well written and presents some impressive results.


# Forward Pass

The math looks very simple, let's try to implement the forward pass in Python:

```python
def batchnorm_forward(x, gamma, beta, eps):
  mu = np.mean(x)
  var = np.var(x)
  xhat = (x - mu) / np.sqrt(var + eps)
  y = gamma * xhat + beta
  return y
```

Easy, right? Now what about the backward pass? Not that easy...

Instead of analytically deriving the formula of the gradient, it is easier to
break the formula into atomic operations where we can directly compute the
gradient, and then use the chain rule to get to the gradient of $$X$$,
$$\gamma$$ and $$\beta$$. This can be represented as a graph, known as the
computational graph, where nodes are mathematical operations and the edges
connect the output of one node to the input of another.

![Computational graph for the Batch Normalization layer]({{site.url}}/img/batch-norm-computational-graph.png)
