---
layout: post
title: Nonparametric Latent Feature Models for Link Prediction
---

This paper is from the ML group in Berkeley.
There is quite a lot of material in non-parametric bayesian;
the reason that I'm going to spend a little bit of time on writing this is because of it's magic approach towards inferring the number of classes at the same time as associating the entities to each class.
I'll wrap up on details later but first I'm going to give enough background for you to follow what is going on in the paper.
From a classical point of view a link prediction task is meant to be defined by a number of edges and a set of nodes where ultimately we aim to predict new possible edgs between nodes.

In ML community it's common to use latent class models to learn latent classes in data as well as associating each node to a number of them.
This apparently simple problem turns out to be very complicated if the number of classes is unknown before observing data. 
Let's say $s_1=$'male high school athletes' and $s_2=$'male high school musician'. 
You have to be able to assign a single class $c=$'high school students' for these examples but through a flow of data you might find this reasonable to split it into two classes $c_1=$'athlete' and $c_2=$'music'. 

Assume $y_{ij}=1$ if there is a link between entity $i$ and $j$ and $y_{i,j}=0$ if not. 
Lets define $Z=[z_{i,j}]\in \{0,1\}^{N\times K}$ the feature value matrix for all examples and $W=[w_{i,j}] \in R^{K\times K}$ the weight matrix.
What a binary feature matrix $Z$ does is to represent data by presence/absence of hidden classes instead of continuous representation.
\begin{equation*}
P(Y|Z,W) = \Pi_{i,j} \Bigg[ P(y_{ij}| Z_i, Z_j,W) = \sigma \big ( Z_i W Z_j^T\big) = \sigma \big( \sum_{k,k'} z_{ik}z_{jk'}w_{kk'}\big) \Bigg ]
\end{equation*}

If I unpack the definition of this function, the $y_{i,j}$ can purely be rendered by the feature vectors and the weights where $\sigma = \frac{1}{1+\exp (-x)}$.
We don't need to invoke very complicated ideas to estimate $P(Y|Z,W)$ but we are about to assume the following priors:
\begin{equation*}
Z \sim IBP(\alpha) 
\end{equation*}
\begin{equation*}
w_{kk'} \sim N(0,\delta^2_w) 
\end{equation*}
\begin{equation*}
y_{ij} \sim \delta(Z_iWZ_j^T)
\end{equation*}

$IBP(.)$ is the indian buffette process and explaining it gives huge amount of insight into this problem.
Let's assume the classes/features as a number of dishes and each customer is assumed to stop by a dish to fill her plate. 
This distribution has a set of very nice properties in particular it captures countable infinite number of classes.
We really don't want our prior somehow magically depends on the size of data.
So the first customer stops by Poisson$(\alpha)$ number of dishes and the $i-$th one take a look at the previously chosen ones based on their popularity and then chooses a new one by Poisson$(\alpha/i)$.
We are going to take the limit of this as $i$ goes to $\infinity$.
This is really cool and really strightforward. 

We need to have data to be able to infere these parameters:
\begin{equation*}
 P(y_{ij}| Z,W,X,\beta) = \sigma \big ( Z_i W Z_j^T + \beta^T X_{ij} \big) 
\end{equation*}

it takes you to compute $X_{i,j} = \exp (-d(i,j))$ which is a scalar similarity metric. 
Therefor from a practical point of view in each step of a sampling process the two most similar nodes should tend to get a link between them. 

This is certainly true based around a generative process but how do we generate these samples.
They proposed an approximate inference via MCMC:
\begin{equations*}
P(z_{ik}=1| Z_{-ik},W,Y) \propto P(Y|z_{ik}=1,Z_{-ik},W)
\end{equations*} 
Just to give you an intuition, the MCMC Gibbs sampling is a method starts with random initialization; 
to be more precise $Z_{-ik}$ is supposed to be able to render the future independant of past.
To be more precise, removing $z_{ik}$ and re-sampling it by $Z_{-ik}$ is effectively operating towards convergance to the real distributin.
We can think exactly analogously to topic modeling where each word is supposed to get a topic label through a large number of documents. 
Take a look at David Blei's dirichlete model for more details.

Hopefully you  picked up this article in less than 7 minutes!

```
@inproceedings{miller:2009,
  title={Nonparametric latent feature models for link prediction},
  author={Miller, Kurt and Jordan, Michael I and Griffiths, Thomas L},
  booktitle={Advances in neural information processing systems},
  pages={1276--1284},
  year={2009}
}
}
``` 

