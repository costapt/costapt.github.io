---
layout:     post
title:      "Efficient Batch Normalization"
subtitle:   "How to implement the Batch Norm layer efficiently"
date:       2016-07-09 18:00:00
author:     "Pedro Costa"
header-img: "img/batch-norm-alt-cover.jpg"
---

In the [previous post]({{site.url}}2016/06/26/batch-norm/) I described the Batch
Normalization layer. I implemented it using the computational graph instead
of using calculus to get a more efficient implementation. In this post I
will derive the batch normalization gradient expression and implement it
in python.

I implemented the Batch Norm layer while doing the assignments of the great
[CS231n](http://cs231n.stanford.edu/) online course, and the full code can be
accessed on my [github](https://github.com/costapt/cs231n).

# Batch Normalization

The Batch Norm layer normalizes its input to have zero mean and unit variance,
just as we usually do to the input data:

$$ \hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma^2_B + \epsilon}} $$

Where $$\mu_B$$ is the per-feature mean of the input batch and $$\sigma^2_B$$
is the variance.
As this operation could limit the representational power of the network, the
authors wanted to make sure that the layer could learn the identity function:

$$ y_i = \gamma \cdot \hat{x}_i + \beta $$

The layer is initialized with $$\gamma = 1$$ and $$\beta = 0$$ in order to
effectively normalize its input. But, when
$$\gamma = \sqrt{\sigma^2_B + \epsilon}$$ and $$\beta = \mu_B$$, the output
of the layer is the same as its input. As such, the layer can learn what's best:
if it should normalize the data to have 0 mean and unit variance, if it should
output the same values as its input or if it should scale and translate the input
with some other values.

For a more in-depth description of the Bach Normalization layer please check my
[previous post]({{site.url}}2016/06/26/batch-norm/) or read the original
[paper](http://arxiv.org/abs/1502.03167).

# Backpropagation

There are two strategies to derive the gradient expression needed to update the
network's weights:

1. Break down the mathematical expressions into atomic operations in order to
build a [computational graph]({{site.url}}2016/06/26/batch-norm/) and make it
easier to use the chain rule.
2. Derive the gradient analytically.

The first strategy offers us a simple abstraction to compute the gradient but
it is usually not the most efficient implementation. By deriving the gradient
expression, it is usually possible to simplify it and remove unnecessary terms.
It turns out that the second strategy yields, indeed, a more efficient gradient
expression for the Batch Norm layer.

Let's start with the simpler parameters: $$\gamma$$ and $$\beta$$. In these cases,
the computational graph already gives us the best possible expression,
but this will help us remember the basics.
Again, we will make use of the chain rule to derive the partial
derivative of the loss with respect to $$\gamma$$:

$$
\begin{equation}
  \frac{\partial l}{\partial \gamma} =
    \sum_i^N \frac{\partial l}{\partial y_i} \cdot \frac{\partial y_i}{\partial \gamma}
\end{equation}
$$

Why do we need to use the chain rule? We have the value of the
$$\frac{\partial l}{\partial y_i}\ $$, as it is provided to us by the next layer,
and we can actually derive  the $$\frac{\partial y_i}{\partial \gamma}\ $$.
That way, we do not need to know anything about the loss function that was used
nor what are the next layers. The gradient expression becomes self-contained,
provided that we are given $$\frac{\partial l}{\partial y_i}\ $$.

Why the summation? $$\gamma$$ is a $$D$$ dimensional vector that is multiplied
by the $$N$$ vectors $$x_i$$ of dimension $$D$$ and so its contributions must be
summed. As I am a very visual person, I better understood this by thinking in
terms of a computational graph:

![Computational graph for the gamma derivative]({{site.url}}/img/batch-norm-gamma-derivative.png)

When we are backpropagating through the computational graph, the gradient
flowing from each $$y_i$$ node arrives at the same $$\gamma$$ node and, as such,
all the gradients must be summed.

The gradient expressions for the partial derivatives of the loss with respect to
$$\gamma$$ and $$\beta$$ become:

$$
\begin{align}
  \frac{\partial l}{\partial \gamma}
    &= \sum_i^N \frac{\partial l}{\partial y_i} \cdot \frac{\partial y_i}{\partial \gamma} \\
    &= \sum_i^N \frac{\partial l}{\partial y_i} \cdot \hat{x_i} \\
    \\
  \frac{\partial l}{\partial \beta}
    &= \sum_i^N \frac{\partial l}{\partial y_i} \cdot \frac{\partial y_i}{\partial \beta} \\
    &= \sum_i^N \frac{\partial l}{\partial y_i}
\end{align}
$$

What we need to compute next is the partial derivative of the loss with respect
to the inputs $$x_i$$, so the previous layers can compute their gradients and
update their parameters. We need to gather all the expressions where $$x_i$$ is
used that has influence in the $$y_i$$ result. Do not forget that:

$$
\begin{align}
  \mu_B &= \frac{1}{N} \sum_i^N x_i \\
  \sigma^2_B &= \frac{1}{N} \sum_i^N (x_i - \mu_B)^2
\end{align}
$$

We can conclude that $$x_i$$ is used to compute $$\hat{x}_i$$, $$\mu_B$$ and
$$\sigma^2$$ and therefore:

$$
\begin{align}
  \frac{\partial l}{\partial x_i}
  &= \frac{\partial l}{\partial \hat{x_i}} \cdot \frac{\partial \hat{x_i}}{\partial x_i} +
     \frac{\partial l}{\partial \sigma^2_B} \cdot \frac{\partial \sigma^2_B}{\partial x_i} +
     \frac{\partial l}{\partial \mu_B} \cdot \frac{\partial \mu_B}{\partial x_i}
\end{align}
$$

Let's compute and simplify each of these terms individually and then we come
back to this expression.

The first term is the easiest one to derive.
Using the same chain rule process as before we arrive to the following expressions:

$$
\begin{align}
  \frac{\partial l}{\partial \hat{x}_i} &= \frac{\partial l}{\partial y_i} \cdot \gamma \\
  \frac{\partial \hat{x_i}}{\partial x_i} &= (\sigma^2_B + \epsilon)^{-\frac{1}{2}} \\
  \frac{\partial l}{\partial \hat{x_i}} \cdot \frac{\partial \hat{x_i}}{\partial x_i}
    &= \frac{\partial l}{\partial y_i} \cdot \gamma \cdot (\sigma^2_B + \epsilon)^{-\frac{1}{2}}
\end{align}
$$

So far so good. The next expressions are a bit longer, but once you get the process of
the chain rule they are just as simple.

$$
\begin{align}
  \frac{\partial l}{\partial \sigma^2_B}
    &= \sum_i^N \frac{\partial l}{\partial \hat{x}_i} \cdot \frac{\partial \hat{x}_i}{\partial \sigma^2_B} \\
    &= \sum_i^N \frac{\partial l}{\partial y_i} \cdot \gamma \cdot (x_i - \mu_B) \cdot (-\frac{1}{2}) \cdot (\sigma^2_B + \epsilon)^{-\frac{3}{2}} \\
    &= -\frac{\gamma \cdot (\sigma^2_B + \epsilon)^{-\frac{3}{2}}}{2} \sum_i^N \frac{\partial l}{\partial y_i} \cdot (x_i - \mu_B)
\end{align}
$$

As what happened with the gradients of $$\gamma$$ and $$\beta$$, to compute
the gradient of $$\sigma^2$$ we need to sum over the contributions of all
elements from the batch. The same happens to the gradient of $$\mu_B$$, as it is
also a $$D$$ dimensional vector:

$$
\begin{align}
  \frac{\partial l}{\partial \mu_B}
  &= \Bigg[\sum_i^N \frac{\partial l}{\partial \hat{x}_i} \cdot \frac{\partial \hat{x}_i}{\partial \mu_B} \Bigg] +
     \Bigg[ \frac{\partial l}{\partial \sigma^2_B} \cdot \frac{\partial \sigma^2_B}{\partial \mu_B} \Bigg] \\
  &= \Bigg[\sum_i^N \frac{\partial l}{\partial y_i} \cdot \gamma \cdot \frac{-1}{\sqrt{\sigma^2_B + \epsilon}} \Bigg] +
     \Bigg[\frac{\partial l}{\sigma^2_B} \cdot \frac{1}{N} \sum_i^N -2(x_i - \mu_B) \Bigg] \\
  &= -\gamma \cdot (\sigma^2_B + \epsilon)^{-\frac{1}{2}} \Big( \sum_i^N \frac{\partial l}{\partial y_i} \Big) -
     \frac{\partial l}{\partial \sigma^2_B} \cdot \frac{2}{N} \Big( \sum_i^N x_i - \mu_B \Big) \\
  &= -\gamma \cdot (\sigma^2_B + \epsilon)^{-\frac{1}{2}} \Big( \sum_i^N \frac{\partial l}{\partial y_i} \Big)
\end{align}
$$

In case you are wondering why I deleted the second term of the expression, it
turns out that $$\sum_i^N x_i - \mu_B$$ is equal to $$0$$.
We are translating all the points so their mean is equal to $$0$$. By looking at
the mean's formula, we can see that the only way for the points to have $$0$$
mean is when their sum is also equal to $$0$$:

$$
\begin{equation}
\sum_i^N ( x_i - \mu_B) = \sum_i^N(x_i) - N \cdot \mu_B = \sum_i^N(x_i) - N \cdot \frac{1}{N} \sum_i^N (x_i) = 0
\end{equation}
$$

Now we can easily compute the rest of the terms:

$$
\begin{align}
  \frac{\partial l}{\partial \sigma^2_B} \cdot \frac{\partial \sigma^2_B}{\partial x_i}
  &= \frac{\partial l}{\partial \sigma^2_B} \cdot \frac{2(x_i - \mu_B)}{N} \\
  &= -\frac{\gamma \cdot (\sigma^2_B + \epsilon)^{-\frac{3}{2}}}{2} \Big( \sum_j^N \frac{\partial l}{\partial y_j} \cdot (x_j - \mu_B) \Big) \cdot \frac{2(x_i - \mu_B)}{N} \\
  &= -\frac{\gamma \cdot (\sigma^2_B + \epsilon)^{-\frac{3}{2}}}{N} \Big( \sum_j^N \frac{\partial l}{\partial y_j} \cdot (x_j - \mu_B) \Big) \cdot (x_i - \mu_B) \\
  &= \frac{\gamma \cdot (\sigma^2_B + \epsilon)^{-\frac{1}{2}}}{N} \Big( \sum_j^N \frac{\partial l}{\partial y_j} \cdot (x_j - \mu_B) \Big) \cdot (x_i - \mu_B) \cdot - (\sigma^2_B + \epsilon)^{-1}
\\
\\
\\
  \frac{\partial l}{\partial \mu_B} \cdot \frac{\partial \mu_B}{\partial x_i}
  &= \frac{\partial l}{\partial \mu_B} \cdot \frac{1}{N} \\
  &= \frac{-\gamma \cdot (\sigma^2_B + \epsilon)^{-\frac{1}{2}}}{N} \Big( \sum_j^N \frac{\partial l}{\partial y_j} \Big)
\end{align}
$$

Merging everything together:

$$
\begin{equation}
\frac{\partial l}{\partial x_i} =
   \frac{\gamma \cdot (\sigma^2_B + \epsilon)^{-\frac{1}{2}}}{N}
      \Bigg[
        N \frac{\partial l}{\partial y_i} -
        \Big( \sum_j^N \frac{\partial l}{\partial y_j} \cdot (x_j - \mu_B) \Big) \cdot (x_i - \mu_B) \cdot (\sigma^2_B + \epsilon)^{-1} +
        \Big( \sum_j^N \frac{\partial l}{\partial y_j} \Big)
      \Bigg]
\end{equation}
$$

Finally we have a single mathematical expression for the partial derivative of
the loss with respect to each input. This is a simpler expression than what
we get by deriving the expression using the
[computational graph]({{site.url}}2016/06/26/batch-norm/).

Translating this to python, we end up with a much more compact method:

```python
def batchnorm_backward_alt(dout, cache):
  gamma, xhat, xcorrected, istd = cache
  N, _ = dout.shape

  dbeta = np.sum(dout, axis=0)
  dgamma = np.sum(xhat * dout, axis=0)
  dx = (gamma * istd / N) * (
          N*dout -
          np.sum(dout * xcorrected, axis=0)*xcorrected*istd**2 -
          np.sum(dout, axis=0))

  return dx, dgamma, dbeta
```

This method is about 2 times faster than the one presented on the previous post.
It might not seem much, but when we are talking about deep neural networks
that may take weeks to train, every little improvement in the end makes a huge
difference.
