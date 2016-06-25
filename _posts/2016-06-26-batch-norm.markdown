---
layout:     post
title:      "Batch Normalization"
subtitle:   "How to implement the Batch Norm layer"
date:       2016-06-26 12:00:00
author:     "Pedro Costa"
header-img: "img/batch-norm-cover.jpg"
---

I have been doing the [CS231n](http://cs231n.stanford.edu/) Stanford's course
assignments when I stumbled upon the Batch Normalization layer. I decided to
write this blog post mainly for myself, so I can revisit everything that I
learned when implementing the forward pass and the backward pass of the
this layer, but I hope it might be of some use to other people.

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
The good news is that it is, but let's see a little
example.

Assume the output of an hidden layer $$X$$ is an $$(N,D)$$ matrix, where
$$N$$ is the number of examples present in the batch and $$D$$ is the number of
hidden units. We start by normalizing $$X$$:

$$ \hat{X} = \frac{X - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}\ ,$$

where $$\mu_B$$ is the mean of the batch, $$\sigma_B^2$$ is the variance and
$$\epsilon$$ is a small number to guarantee that there won't be any division
by 0.
As $$\mu_B$$ and $$\sigma_B^2$$ are differentiable, we can see that $$ \hat{X}$$
is also differentiable, and so we are good to go.

Then, the authors did something really clever. They realized that by
normalizing the output of a layer they could limit its representational power
and, therefore, they wanted to make sure that the Batch Norm layer could fall
back to the identity function.

$$ y = \gamma \hat{X} + \beta $$

Note that when $$\gamma = \sqrt{\sigma_B^2 + \epsilon}$$ and $$\beta = \mu_B$$
the Batch Norm simply outputs the previous layer's activations.
The network has the ability to ignore the Batch Norm layer if that is the
optimal thing to do.
As you might have guessed by now, $$\gamma$$ and $$\beta$$ are learnable
parameters that are initialized with $$\gamma = 1$$ and $$\beta = 0$$.

The authors claim that Batch Norm has several nice properties such as:

- **Reduce internal covariate shift**:
[Ioffe et al, 2015](http://arxiv.org/abs/1502.03167) define internal covariate
shift as "the change in the distribution of the network activations due to the
change in network parameters during training".
Imagine how hard it is to learn a model when the inputs are always
changing. With Batch Norm, the next layer can expect the same input
distribution at each iteration.
- **Regularization effect**: It is claimed that with Batch Norm we could
disregard or reduce the strength of Dropout. The value given to one training
example depends on the other examples in the batch, since the mean and variance
are computed on the batch level and, therefore, it is not deterministic. This
property can increase the generalization of the network.
- **Reduce probability of vanishing or exploding gradients**: As the Batch Norm
layer is placed prior to the non-linearity, it prevents the training to get
stuck on saturated areas of non-linearities, solving the issue of vanishing
gradients. Also, it makes the network more robust to the scale of the
parameters, solving the issue of exploding gradients.

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

$$ \frac{\partial l}{\partial \beta} = \frac{\partial l}{\partial y} * \frac{\partial y}{\partial \beta} $$

$$\frac{\partial l}{\partial y}$$ tells us how the loss of the entire network
would grow/decrease if the output $$y$$ would increase a tiny amount.
This value is computed by the layer that is on top of the Batch Norm and so
it is given to us as an input of the backward pass.
Now we need to compute $$\frac{\partial y}{\partial \beta}$$ :

$$ \frac{\partial y}{\partial \beta} = \frac{\partial (\hat{X} * \gamma + \beta)}{\partial \beta} = 1$$

If you increase $$\beta$$ by a tiny amount $$h$$, then $$y$$ is expected to
become $$y + 1 * h$$ as well. That makes sense! But what about the loss?

$$ \frac{\partial l}{\partial \beta} = \frac{\partial l}{\partial y} * 1 = \frac{\partial l}{\partial y} $$

So the gradient of $$\beta$$ is simply the gradient that reaches the network.
But wait, the dimension of $$\beta$$ is not the same as the dimension of $$y$$!
Something is wrong, right?

In truth, it is not possible to sum two matrices with different sizes. This
line `y = xhat * gamma + beta` should not work in the first place. What numpy
is doing behind the scenes is called broadcasting. In this case, it will simply
add $$\beta$$ to each line of the other matrix (it does the same thing to
multiply $$\gamma$$ and $$\hat{X}$$). To arrive to the $$(D,)$$ dimensional
array $$\frac{\partial l}{\partial \beta}$$  we just need to sum each line
of $$\frac{\partial l}{\partial y}$$ :

```python
  dbeta = np.sum(dy, axis=0)
```

Here are the remaining partial derivatives:

1. $$ \frac{\partial l}{\partial \gamma} =
\frac{\partial l}{\partial y} * \frac{\partial y}{\partial \gamma} =
\frac{\partial l}{\partial y} * \hat{X} $$

2. $$ \frac{\partial l}{\partial \hat{X}} =
\frac{\partial l}{\partial y} * \frac{\partial y}{\partial \hat{X}} =
\frac{\partial l}{\partial y} * \gamma $$

3. $$ \frac{\partial l}{\partial istd} =
\frac{\partial l}{\partial \hat{X}} * \frac{\partial \hat{X}}{\partial istd} =
\frac{\partial l}{\partial \hat{X}} * (X - \mu_B) $$

4. $$ \frac{\partial l}{\partial \sigma} =
\frac{\partial l}{\partial istd} * \frac{\partial istd}{\partial \sigma} =
\frac{\partial l}{\partial istd} * \frac{-1}{\sigma^2}$$

5. $$ \frac{\partial l}{\partial \sigma^2} =
\frac{\partial l}{\partial \sigma} * \frac{\partial \sigma}{\partial \sigma^2} = \frac{\partial l}{\partial \sigma} * \frac{1}{2\sigma} $$

6. $$ \frac{\partial l}{\partial xsquarred} =
\frac{\partial l}{\partial \sigma^2} * \frac{\partial \sigma^2}{\partial xsquarred} =
\frac{\partial l}{\partial \sigma^2} * \frac{1}{N}$$

7. $$ \frac{\partial l}{\partial xcorrected} =
\frac{\partial l}{\partial \hat{X}} * \frac{\partial \hat{X}}{\partial xcorrected} + \frac{\partial l}{\partial xsquarred} * \frac{\partial xsquarred}{\partial xcorrected} = \\
= \frac{\partial l}{\partial \hat{X}} * \frac{1}{\sigma} +
\frac{\partial l}{\partial xsquarred} * 2 * xcorrected $$

8. $$ \frac{\partial l}{\partial \mu} =
\frac{\partial l}{\partial xcorrected} * \frac{\partial xcorrected}{\partial \mu} =
\frac{\partial l}{\partial xcorrected} * -1$$

9. $$ \frac{\partial l}{\partial X} =
\frac{\partial l}{\partial xcorrected} * \frac{\partial xcorrected}{\partial X} +
\frac{\partial l}{\partial \mu} * \frac{\partial \mu}{\partial X} = \\
= \frac{\partial l}{\partial xcorrected} * 1 +
\frac{\partial l}{\partial \mu} * \frac{1}{N} $$

Just as when we are doing the forward pass we use output of the one
node the input of another node, in the backward pass we use the gradient of
one node to compute the gradient of the previous one. It might be clearer in
python:

```python
def batchnorm_backward(dldy, cache):
  gamma, xhat, xcorrected, istd, std = cache
  N, D = dldy.shape

  # Compute the gradient of beta and gamma. Both with (D,) dimension
  dbeta = np.sum(dldy, axis=0)
  dgamma = np.sum(xhat * dout, axis=0)

  # Equation 2 (N,D)
  dxhat = dldy * gamma

  # Equation 3 (D,)
  distd = np.sum(dxhat * xcorrected, axis=0)

  # Equation 4 (D,)
  dstd = distd * -1 / (std**2)

  # Equation 5 (D,)
  dvar = dstd * 1 / (2 * std)

  # Equation 6 (N,D)
  dxsquarred = dvar * np.ones((N, D)) / N

  # Equation 7 (N,D)
  dxcorrected = dxhat * istd + dxsquarred * 2 * xcorrected

  # Equation 8 (D,)
  dmu = np.sum(dxcorrected, axis=0) * -1

  # Equation 9 (N,D)
  dx = dxcorrected * 1 + dmu * np.ones((N, D)) / N

  return dx, dgamma, dbeta
```

That's it, we successfully implemented the forward and backward pass of the
Batch Norm layer.

# Test time

While it is acceptable to compute the mean and variance on a mini-batch
when we are training the network, the same does not hold on test time.
When the batch is large enough, its mean and variance will be close to the
population's and, as previously stated, the non-deterministic mean and variance
has regularizing effects.

On test time, we want the model to be deterministic. The classification result
of an image does not depend on the images that we are feeding the model to be
classified at the same time.
Also, we want to be able to run the model on a single example and it does not
make sense to use its mean and variance (the variance would be 0).
In conclusion, we need a fixed $$\mu$$ and $$\sigma^2$$ that can be used during
inference.

To solve this issue, the authors proposed to compute these values using the
population instead of the mini-batch statistics. Since we need to consider
the training examples as a sample from all the possible examples, the authors
use the mean and the unbiased variance over the mini-batches:

$$ \mu = \frac{1}{B} \sum_i^B \mu_i $$

$$ \sigma^2 = \frac{1}{B - 1} \sum_i^B \sigma^2_i $$

Where $$B$$ is the number of mini-batches, $$\mu_i$$ is the mean of the
mini-batch $$i$$ and $$\sigma^2_i$$ is the variance.

The authors also proposed to take a moving average of the mean and variance to
track the accuracy of the model as it trains:

$$ \mu = m * \mu + (1 - m) * \mu_i $$

$$ \sigma^2 = m * \sigma^2 + (1 - m) * \sigma^2_i $$

Where $$m$$ is the momentum in the range $$[0, 1[$$ and is typically a value
close to 1 (i.e. $$0.9$$). In fact, we can use the results of the moving
average as our global statistics instead of using the previous more complicated
method. This was the implementation choice made by the [Keras](http://keras.io/)
library and what was suggested by the [CS231n](http://cs231n.stanford.edu/)
course and, as such, that is the method that we are going to implement here.

So the forward pass becomes:

```python
def batchnorm_forward(x, gamma, beta, eps, momentum, moving_mean, moving_var, mode):

  y, cache = None, None
  if mode == 'train':
    mu = np.mean(x, axis=0)
    xcorrected = x - mu
    xsquarred = xcorrected**2
    var = np.mean(xsquarred, axis=0)
    std = np.sqrt(var + eps)
    istd = 1 / std
    xhat = xcorrected * istd
    y = xhat * gamma + beta

    # Compute the moving average of the mean and variance
    moving_mean = momentum * moving_mean + (1 - momentum) * mu
    moving_var = momentum * moving_var + (1 - momentum) * var

    cache = (gamma, xhat, xcorrected, istd, std)
  elif mode == 'test':
    # Use the moving mean and variance on test time
    std = np.sqrt(moving_var + eps)
    y = (x - moving_mean) / std
    y = gamma * out + beta
  else:
    raise ValueError('Invalid forward batchnorm mode.')

  return y, cache, moving_mean, moving_var
```

And that's it! We have a fully functioning Batch Normalization layer. However,
this is the naive implementation. We relied on the computational graph to
derive the gradient expressions, although we could use calculus to derive a
more efficient expression. I may derive that on a future post.

**Note:** this is my first blog post, and it probably contains some errors.
If you find any or if you have any advice on how to get better, please send
me an email. I will be appreciated.


[//]: # (# Improving the Backward Pass)

[//]: # (While relying on the computational graph and the chain rule to compute the gradient of $$X$$, $$\gamma$$ and $$\beta$$ with respect to the loss is the easiest and more intuitive way of reaching to a solution, it is not the most efficient one. It would be much better if we just used one equation to get the value of the gradient of $$X$$ instead of using 8.)

[//]: # (Since each equation that we derived uses a value computed on the previous one, we can write one big equation for $$\frac{\partial l}{\partial X}$$ and then simplify it. For example, on equation 9 we can replace the $$\frac{\partial l}{\partial \mu}$$ by $$\frac{\partial l}{\partial xcorrected} * -1$$ and do that recursively until we get one equation that is only a function of $$\frac{\partial l}{\partial y}$$ .)
