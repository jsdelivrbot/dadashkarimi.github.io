---
layout: post
title: Nonparametric Latent Feature Models for Link Prediction 8
---

This paper is from the ML group in Berkeley.
There is quite a lot of material in non-parametric bayesian;
the reason that I'm going to spend a little bit of time on writing this is because of it's magic approach towards inferring the number of classes at the same time as associating the entities to each class.
I'll wrap up on details later but first I'm going to give enough background for you to follow what is going on in the paper.
From a classical point of view a link prediction task is meant to be defined by a number of edges and a set of nodes where ultimately we aim to predict new possible edgs between nodes.

In ML community it's common to use latent class models to learn latent classes in data as well as associating each node to a number of them.
This apparently simple problem turns out to be very complicated if the number of classes is unknown before observing data. 
Let's say <img alt="$s_1=$" style="position:relative; top:7px;" src="https://rawgit.com/dadashkarimi/dadashkarimi.github.io/master/svgs/e63d5448ef61e2fd17edd20e83eeed26.svg?sanitize=true"/>'male high school athletes' and <img alt="$s_2=$" style="position:relative; top:7px;" src="https://rawgit.com/dadashkarimi/dadashkarimi.github.io/master/svgs/caa80954b35018e0be0ea8d2119017fd.svg?sanitize=true"/>'male high school musician'. 
You have to be able to assign a single class <img alt="$c=$" style="position:relative; top:7px;" src="https://rawgit.com/dadashkarimi/dadashkarimi.github.io/master/svgs/3318bc78ce112b6761f73b9288905746.svg?sanitize=true"/>'high school students' for these examples but through a flow of data you might find this reasonable to split it into two classes <img alt="$c_1=$" style="position:relative; top:7px;" src="https://rawgit.com/dadashkarimi/dadashkarimi.github.io/master/svgs/b5cbeca3815c7e70bd9ff3164e0e51ee.svg?sanitize=true"/>'athlete' and <img alt="$c_2=$" style="position:relative; top:7px;" src="https://rawgit.com/dadashkarimi/dadashkarimi.github.io/master/svgs/3367c6d79b877c913dccd683f3951fb9.svg?sanitize=true"/>'music'. 

Assume <img alt="$y_{ij}=1$" style="position:relative; top:7px;" src="https://rawgit.com/dadashkarimi/dadashkarimi.github.io/master/svgs/193959917d5e875406dd5eab26c8139e.svg?sanitize=true"/> if there is a link between entity <img alt="$i$" style="position:relative; top:2px;" src="https://rawgit.com/dadashkarimi/dadashkarimi.github.io/master/svgs/77a3b857d53fb44e33b53e4c8b68351a.svg?sanitize=true"/> and <img alt="$j$" style="position:relative; top:2px;" src="https://rawgit.com/dadashkarimi/dadashkarimi.github.io/master/svgs/36b5afebdba34564d884d347484ac0c7.svg?sanitize=true"/> and <img alt="$y_{i,j}=0$" style="position:relative; top:7px;" src="https://rawgit.com/dadashkarimi/dadashkarimi.github.io/master/svgs/67e3ad425f262d5d43ef11d4da43e404.svg?sanitize=true"/> if not. 
Lets define <img alt="$Z=[z_{i,j}]\in \{0,1\}^{N\times K}$" style="position:relative; top:7px;" src="https://rawgit.com/dadashkarimi/dadashkarimi.github.io/master/svgs/859d4e2097e2da22e1254770851ac6e3.svg?sanitize=true"/> the feature value matrix for all examples and <img alt="$W=[w_{i,j}] \in R^{K\times K}$" style="position:relative; top:7px;" src="https://rawgit.com/dadashkarimi/dadashkarimi.github.io/master/svgs/daa1032b51643b282c7bcad307157bbf.svg?sanitize=true"/> the weight matrix.
What a binary feature matrix <img alt="$Z$" style="position:relative; top:7px;" src="https://rawgit.com/dadashkarimi/dadashkarimi.github.io/master/svgs/5b51bd2e6f329245d425b8002d7cf942.svg?sanitize=true"/> does is to represent data by presence/absence of hidden classes instead of continuous representation.
<p align="center"><img alt="\begin{equation*}&#10;P(Y|Z,W) = \Pi_{i,j} \Bigg[ P(y_{ij}| Z_i, Z_j,W) = \sigma \big ( Z_i W Z_j^T\big) = \sigma \big( \sum_{k,k'} z_{ik}z_{jk'}w_{kk'}\big) \Bigg ]&#10;\end{equation*}" src="https://rawgit.com/dadashkarimi/dadashkarimi.github.io/master/svgs/955edbc0b2eb83e8fc2d93a66ca13a71.svg?sanitize=true" align="middle" width="523.52355pt" height="50.765715pt"/></p>

If I unpack the definition of this function, the <img alt="$y_{i,j}$" style="position:relative; top:2px;" src="https://rawgit.com/dadashkarimi/dadashkarimi.github.io/master/svgs/782a78d8c11a2145d873d3bc48870864.svg?sanitize=true"/> can purely be rendered by the feature vectors and the weights where <img alt="$\sigma = \frac{1}{1+\exp (-x)}$" style="position:relative; top:7px;" src="https://rawgit.com/dadashkarimi/dadashkarimi.github.io/master/svgs/9eaa22a843a8020f1a347b764412b390.svg?sanitize=true"/>.
We don't need to invoke very complicated ideas to estimate <img alt="$P(Y|Z,W)$" style="position:relative; top:7px;" src="https://rawgit.com/dadashkarimi/dadashkarimi.github.io/master/svgs/e1ce5fa1d74b30b52d79c04d7177e556.svg?sanitize=true"/> but we are about to assume the following priors:
<p align="center"><img alt="\begin{equation*}&#10;Z \sim IBP(\alpha) &#10;\end{equation*}" src="https://rawgit.com/dadashkarimi/dadashkarimi.github.io/master/svgs/18c63f10b9a4ddca28b1e6bc16712caf.svg?sanitize=true" align="middle" width="92.033205pt" height="16.376943pt"/></p>
<p align="center"><img alt="\begin{equation*}&#10;w_{kk'} \sim N(0,\delta^2_w) &#10;\end{equation*}" src="https://rawgit.com/dadashkarimi/dadashkarimi.github.io/master/svgs/4a56963c4a4f9bfe3637530d80b56ac5.svg?sanitize=true" align="middle" width="114.4803pt" height="18.269295pt"/></p>
<p align="center"><img alt="\begin{equation*}&#10;y_{ij} \sim \delta(Z_iWZ_j^T)&#10;\end{equation*}" src="https://rawgit.com/dadashkarimi/dadashkarimi.github.io/master/svgs/5568287b2ca605e21e193a3d04493aea.svg?sanitize=true" align="middle" width="119.228505pt" height="20.913915pt"/></p>

<img alt="$IBP(.)$" style="position:relative; top:7px;" src="https://rawgit.com/dadashkarimi/dadashkarimi.github.io/master/svgs/cf8b75d677c758fa8caf59ef55c91a57.svg?sanitize=true"/> is the indian buffette process and explaining it gives huge amount of insight into this problem.
Let's assume the classes/features as a number of dishes and each customer is assumed to stop by a dish to fill her plate. 
This distribution has a set of very nice properties in particular it captures countable infinite number of classes.
We really don't want our prior somehow magically depends on the size of data.
So the first customer stops by Poisson<img alt="$(\alpha)$" style="position:relative; top:7px;" src="https://rawgit.com/dadashkarimi/dadashkarimi.github.io/master/svgs/38f17d4708f8ad119c90e54b43954d60.svg?sanitize=true"/> number of dishes and the <img alt="$i-$" style="position:relative; top:2px;" src="https://rawgit.com/dadashkarimi/dadashkarimi.github.io/master/svgs/7b7ccf0dc7f33e23877ead84bb57af55.svg?sanitize=true"/>th one take a look at the previously chosen ones based on their popularity and then chooses a new one by Poisson<img alt="$(\alpha/i)$" style="position:relative; top:7px;" src="https://rawgit.com/dadashkarimi/dadashkarimi.github.io/master/svgs/a4ad10c795082c8b01669762c3919e17.svg?sanitize=true"/>.
We are going to take the limit of this as <img alt="$i$" style="position:relative; top:2px;" src="https://rawgit.com/dadashkarimi/dadashkarimi.github.io/master/svgs/77a3b857d53fb44e33b53e4c8b68351a.svg?sanitize=true"/> goes to <img alt="$\infinity$" style="position:relative; top:7px;" src="https://rawgit.com/dadashkarimi/dadashkarimi.github.io/master/svgs/e0c07b834cc98cd01db854cdce833d2d.svg?sanitize=true"/>.
This is really cool and really strightforward. 

We need to have data to be able to infere these parameters:
<p align="center"><img alt="\begin{equation*}&#10; P(y_{ij}| Z,W,X,\beta) = \sigma \big ( Z_i W Z_j^T + \beta^T X_{ij} \big) &#10;\end{equation*}" src="https://rawgit.com/dadashkarimi/dadashkarimi.github.io/master/svgs/6170744f6ba3a56c8f93a068152b977b.svg?sanitize=true" align="middle" width="291.73155pt" height="20.913915pt"/></p>

it takes you to compute <img alt="$X_{i,j} = \exp (-d(i,j))$" style="position:relative; top:7px;" src="https://rawgit.com/dadashkarimi/dadashkarimi.github.io/master/svgs/21307b65cad84cae9442087ddc854d77.svg?sanitize=true"/> which is a scalar similarity metric. 
Therefor from a practical point of view in each step of a sampling process the two most similar nodes should tend to get a link between them. 

This is certainly true based around a generative process but how do we generate these samples.
They proposed an approximate inference via MCMC:
<p align="center"><img alt="\begin{equations*}&#10;P(z_{ik}=1| Z_{-ik},W,Y) \propto P(Y|z_{ik}=1,Z_{-ik},W)&#10;\end{equations*}" src="https://rawgit.com/dadashkarimi/dadashkarimi.github.io/master/svgs/2c2ca4e05835e160ee94bab2c00b3772.svg?sanitize=true" align="middle" width="337.4976pt" height="16.376943pt"/></p> 
Just to give you an intuition, the MCMC Gibbs sampling is a method starts with random initialization; 
to be more precise <img alt="$Z_{-ik}$" style="position:relative; top:7px;" src="https://rawgit.com/dadashkarimi/dadashkarimi.github.io/master/svgs/5e106c28ccbc3936410489fe9da8d97a.svg?sanitize=true"/> is supposed to be able to render the future independant of past.
To be more precise, removing <img alt="$z_{ik}$" style="position:relative; top:2px;" src="https://rawgit.com/dadashkarimi/dadashkarimi.github.io/master/svgs/e444b929a98b82edf6fd7171777f06bf.svg?sanitize=true"/> and re-sampling it by <img alt="$Z_{-ik}$" style="position:relative; top:7px;" src="https://rawgit.com/dadashkarimi/dadashkarimi.github.io/master/svgs/5e106c28ccbc3936410489fe9da8d97a.svg?sanitize=true"/> is effectively operating towards convergance to the real distributin.
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

