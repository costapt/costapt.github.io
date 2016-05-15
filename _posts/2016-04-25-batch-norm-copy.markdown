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
connect the output of one node to the input of another:

![Computational graph for the Batch Normalization layer]({{site.url}}/img/batch-norm-computational-graph.png)

We follow the dark arrows for the forward pass and then we backpropagate the
error using the red ones. The dimension of the output of each node is displayed
on top of each node. For instance, $$X$$ is an $$(N, D)$$ matrix and the output
of the node that computes the mean of each dimension of $$X$$ is a vector with
$$D$$ elements.

The forward pass then becomes:

```python
def batchnorm_forward(x, gamma, beta, eps):
  # Step 1
  mu = np.mean(x, axis=0)

  # Step 2
  xcorrected = x - mu

  # Step 3
  xsquarred = xcorrected**2

  # Step 4
  var = np.mean(xsquarred, axis=0)

  # Step 5
  std = np.sqrt(var + eps)

  # Step 6
  istd = 1 / std

  # Step 7
  xhat = xcorrected * istd

  # Step 8 and 9
  y = xhat * gamma + beta

  # Store some variables that will be needed for the backward pass
  cache = (gamma, xhat, xcorrected, istd, std)

  return y, cache
```

# Backward Pass

How do we get the gradient of $$X$$, $$\gamma$$ and $$\beta$$ with respect to
the loss $$l$$? We use the chain rule to transverse the computational graph on the
opposite direction (red arrows).

Let's start with the computation of the $$\frac{\partial l}{\partial \beta}$$
to get an idea of how it is done:

$$ \frac{\partial l}{\partial \beta} = \frac{\partial y}{\partial \beta} * \frac{\partial l}{\partial y} $$

Let's say that $$\frac{\partial l}{\partial y}$$ is given to us as input. It
tells us how the loss of the entire network would grow/decrease if the output
$$y$$ would increase a tiny amount. Now we need to compute
$$\frac{\partial y}{\partial \beta}$$:

$$ \frac{\partial y}{\partial \beta} = \frac{\partial (\hat{X} * \gamma + \beta)}{\partial \beta} = 1$$

If you increase $$\beta$$ by a tiny amount $$h$$, then $$y$$ is expected to
become $$y + h$$ as well. That makes sense! But what about the loss?

$$ \frac{\partial l}{\partial \beta} = 1 * \frac{\partial l}{\partial y} = \frac{\partial l}{\partial y} $$

So the gradient of $$\beta$$ is simply the gradient that reaches the network.
But wait, the dimension of $$\beta$$ is not the same as the dimension of $$y$$!
Something is wrong, right? Not quite.

In truth, it is not possible to sum two matrices with different sizes. This
line `y = xhat * gamma + beta` should not work in the first place. What numpy
is doing behind the scene is called broadcasting. In this case, it will simply
add $$\beta$$ to each line of the other matrix (it does the same thing to
multiply $$\gamma$$ and $$\hat{X}$$). To arrive to the $$(D,)$$ dimensional
array $$\frac{\partial l}{\partial \beta}$$ we just need to sum each line
of $$\frac{\partial l}{\partial y}$$. Or in code:

```python
  dbeta = np.sum(dy, axis=0)
```

Here are the remaining partial derivatives:

$$ \frac{\partial l}{\partial \gamma} = \frac{\partial y}{\partial \gamma} * \frac{\partial l}{\partial y} = \hat{X} * \frac{\partial l}{\partial y} $$

$$ \frac{\partial l}{\partial \hat{X}} = \frac{\partial y}{\partial \hat{X}} * \frac{\partial l}{\partial y} = \gamma * \frac{\partial l}{\partial y} $$
