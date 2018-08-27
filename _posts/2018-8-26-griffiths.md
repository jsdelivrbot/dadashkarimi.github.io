---
layout: post
title: Nonparametric Latent Feature Models for Link Prediction
---

This paper is from the ML group in Berkeley.
There is quite a lot of material in non-parametric bayesian;
the reason that I'm going to spend a little bit of time on writing this is because of it's magic approach towards inferring the number of classes at the same time as associating the entities to each.
I'll wrap up on details later but first I'm going to give enough background for you to follow what is going on in the paper.
From a classical point of view a link prediction task is meant to be defined by a finite number of classes and a set of nodes.
This apparently simple problem turns out to be very complicated if the number of classis is unknown before observing data. 
Let's say <img alt="$s_1=$" style="position:relative; top:7px;" src="https://rawgit.com/dadashkarimi/dadashkarimi.github.io/master/svgs/e63d5448ef61e2fd17edd20e83eeed26.svg?sanitize=true"/>'male high school athletes' and <img alt="$s_2=$" style="position:relative; top:7px;" src="https://rawgit.com/dadashkarimi/dadashkarimi.github.io/master/svgs/caa80954b35018e0be0ea8d2119017fd.svg?sanitize=true"/>'male high school musician'. 
You have to be able to assign a single class <img alt="$c=$" style="position:relative; top:7px;" src="https://rawgit.com/dadashkarimi/dadashkarimi.github.io/master/svgs/3318bc78ce112b6761f73b9288905746.svg?sanitize=true"/>'high school students' for these examples but through a flow of data you might find this reasonable to split it into two classes <img alt="$c_1=$" style="position:relative; top:7px;" src="https://rawgit.com/dadashkarimi/dadashkarimi.github.io/master/svgs/b5cbeca3815c7e70bd9ff3164e0e51ee.svg?sanitize=true"/>'athlete' and <img alt="$c_2=$" style="position:relative; top:7px;" src="https://rawgit.com/dadashkarimi/dadashkarimi.github.io/master/svgs/3367c6d79b877c913dccd683f3951fb9.svg?sanitize=true"/>'music'. 

Assume <img alt="$y_{ij}=1$" style="position:relative; top:7px;" src="https://rawgit.com/dadashkarimi/dadashkarimi.github.io/master/svgs/193959917d5e875406dd5eab26c8139e.svg?sanitize=true"/> if there is a link between entity <img alt="$i$" style="position:relative; top:2px;" src="https://rawgit.com/dadashkarimi/dadashkarimi.github.io/master/svgs/77a3b857d53fb44e33b53e4c8b68351a.svg?sanitize=true"/> and <img alt="$j$" style="position:relative; top:2px;" src="https://rawgit.com/dadashkarimi/dadashkarimi.github.io/master/svgs/36b5afebdba34564d884d347484ac0c7.svg?sanitize=true"/> and <img alt="$y_{i,j}=0$" style="position:relative; top:7px;" src="https://rawgit.com/dadashkarimi/dadashkarimi.github.io/master/svgs/67e3ad425f262d5d43ef11d4da43e404.svg?sanitize=true"/> if not. 
Lets define <img alt="$Z=[z_{i,j}]\in {0,1}^{N\times K}$" style="position:relative; top:7px;" src="https://rawgit.com/dadashkarimi/dadashkarimi.github.io/master/svgs/4ee8061a5761da701621649613786b8a.svg?sanitize=true"/> the feature value matrix for all examples and <img alt="$W=[w_{i,j}] \in R^{K\times K}$" style="position:relative; top:7px;" src="https://rawgit.com/dadashkarimi/dadashkarimi.github.io/master/svgs/daa1032b51643b282c7bcad307157bbf.svg?sanitize=true"/> the weight matrix.

<p align="center"><img alt="\begin{equation*}&#10;Pr(Y|Z,W) = \Pi_{i,j} \Bigg[ Pr(y_{ij}| Z_i, Z_j,W) = \sigma \big ( Z_i W Z_j^T\big) = \sigma \big( \sum_{k,k'} z_{ik}z_{jk'}w_{kk'}\big) \Bigg ]&#10;\end{equation*}" src="https://rawgit.com/dadashkarimi/dadashkarimi.github.io/master/svgs/03f12488dd55351438187941a5cd980d.svg?sanitize=true" align="middle" width="539.21505pt" height="50.765715pt"/></p>

If I unpack the definition of this function, the <img alt="$y_{i,j}$" style="position:relative; top:2px;" src="https://rawgit.com/dadashkarimi/dadashkarimi.github.io/master/svgs/782a78d8c11a2145d873d3bc48870864.svg?sanitize=true"/> is purely defined by features and the weights where <img alt="$\sigma = \frac{1}{1+\exp (-x)}$" style="position:relative; top:7px;" src="https://rawgit.com/dadashkarimi/dadashkarimi.github.io/master/svgs/9eaa22a843a8020f1a347b764412b390.svg?sanitize=true"/>.
We don't need to invoke very complicated ideas but we are about to assume the following priors:
<p align="center"><img alt="\begin{equation*}&#10;Z \sim IBP(\alpha) &#10;\end{equation*}" src="https://rawgit.com/dadashkarimi/dadashkarimi.github.io/master/svgs/18c63f10b9a4ddca28b1e6bc16712caf.svg?sanitize=true" align="middle" width="92.033205pt" height="16.376943pt"/></p>
<p align="center"><img alt="\begin{equation*}&#10;w_{kk'} \sim N(0,\delta^2_w) &#10;\end{equation*}" src="https://rawgit.com/dadashkarimi/dadashkarimi.github.io/master/svgs/4a56963c4a4f9bfe3637530d80b56ac5.svg?sanitize=true" align="middle" width="114.4803pt" height="18.269295pt"/></p>
<p align="center"><img alt="\begin{equation*}&#10;y_{ij} \sim \delta(Z_iWZ_j^T)&#10;\end{equation*}" src="https://rawgit.com/dadashkarimi/dadashkarimi.github.io/master/svgs/5568287b2ca605e21e193a3d04493aea.svg?sanitize=true" align="middle" width="119.228505pt" height="20.913915pt"/></p>

explaining <img alt="$IBP(.)$" style="position:relative; top:7px;" src="https://rawgit.com/dadashkarimi/dadashkarimi.github.io/master/svgs/cf8b75d677c758fa8caf59ef55c91a57.svg?sanitize=true"/> or indian buffette process gives huge amount of insight into this problem.
Let's assume the classes as a number of dishes and each customer is assumed to stop by a dish to fill her plate. 
This distribution has a set of very nice properties in particular it captures countable infinite number of classes.
We really don't want our prior somehow magically depends on the size of data.
So the first customer stops by <img alt="$Poisson(\alpha)$" style="position:relative; top:7px;" src="https://rawgit.com/dadashkarimi/dadashkarimi.github.io/master/svgs/aca8c8df07e723a50f74fb84355dea28.svg?sanitize=true"/> number of dishes and the <img alt="$i-$" style="position:relative; top:2px;" src="https://rawgit.com/dadashkarimi/dadashkarimi.github.io/master/svgs/7b7ccf0dc7f33e23877ead84bb57af55.svg?sanitize=true"/>th one take a look at the previously chosen ones based on their popularity and then chooses a new one by <img alt="$Poisson(\alpha/i)$" style="position:relative; top:7px;" src="https://rawgit.com/dadashkarimi/dadashkarimi.github.io/master/svgs/65a0b92e014fbbb7c86696a380ce0b08.svg?sanitize=true"/>.
We are going to take the limit of this as <img alt="$i$" style="position:relative; top:2px;" src="https://rawgit.com/dadashkarimi/dadashkarimi.github.io/master/svgs/77a3b857d53fb44e33b53e4c8b68351a.svg?sanitize=true"/> goes to <img alt="$\infinity$" style="position:relative; top:7px;" src="https://rawgit.com/dadashkarimi/dadashkarimi.github.io/master/svgs/e0c07b834cc98cd01db854cdce833d2d.svg?sanitize=true"/>.
This is really cool and really strightforward. 

It gives a huge amount of insight into
this is certainly true in ..
we are gonna take this simple idea and generalize it ..
we realy don't want our prior somehowe magically depends on 
it has some set of very nice properties
we are going to take the limit of this as go goes to infin
naturally it captures this fact that
high weight between them
should tend to have same labels
if I unpack the definition of function 
is purely defined in terms of 
the weighted average of its neighbours 
that's what this equation says
this is really cool, really simple
If I clam a few nodes, assuming that the graph is connected
a and b have been widely studied in whole bunch of different fields
I'll come to that point later
it takes you 12 points to reach that level 

I hope you pick this article up in less than 7 minutes!

```
@inproceedings{miller2009nonparametric,
  title={Nonparametric latent feature models for link prediction},
  author={Miller, Kurt and Jordan, Michael I and Griffiths, Thomas L},
  booktitle={Advances in neural information processing systems},
  pages={1276--1284},
  year={2009}
}
}
``` 

