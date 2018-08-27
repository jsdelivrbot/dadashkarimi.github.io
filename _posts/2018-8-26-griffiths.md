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
Let's say <img alt="$s1=$" style="position:relative; top:7px;" src="https://rawgit.com/dadashkarimi/dadashkarimi.github.io/master/svgs/1ad44539d78f517246cd91da0b5f120a.svg?sanitize=true"/>'male high school athletes' and <img alt="$s2=$" style="position:relative; top:7px;" src="https://rawgit.com/dadashkarimi/dadashkarimi.github.io/master/svgs/c2e2b87c9e4fa1ecf1e0d832adf6a6b3.svg?sanitize=true"/>'male high school musician'. 
You have to be able to assign a single class <img alt="$c=$" style="position:relative; top:7px;" src="https://rawgit.com/dadashkarimi/dadashkarimi.github.io/master/svgs/3318bc78ce112b6761f73b9288905746.svg?sanitize=true"/>'high school students' for these examples but through a flow of data you might find this reasonable to split it into two classes <img alt="$c1=$" style="position:relative; top:7px;" src="https://rawgit.com/dadashkarimi/dadashkarimi.github.io/master/svgs/0c0f6640cdf3c9eb60c91890e065cafc.svg?sanitize=true"/>'athlete' and <img alt="$c2=$" style="position:relative; top:7px;" src="https://rawgit.com/dadashkarimi/dadashkarimi.github.io/master/svgs/d92d11a83b658cadbb4782837a91f7b1.svg?sanitize=true"/>'music'. 


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

