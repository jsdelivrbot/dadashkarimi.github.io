---
layout: post
title: Zero-shot Transfer Learning for Semantic Parsing
---

This paper is one of my memorabale works in LILY group particularly with Alexander Fabbri.
In 2017 in John Lafferty's non-parametric estimation course we were introduced to the award winning paper by Percy Liang's group. 
There is quit a lot of material in deep learning but there is not really that much works openning this black box. 
Inspired by 'Understanding Black-box Predictions via Influence Functions' we started expanding our running works on semantic parsing. 

In this blog post we don't need to invoke any complicated ideas but i'mg going to give enough background for you to follow the rest. 
In classical definition semantic parsing is defined by a task generating logical forms of a sequence of a human language. 
You can think of this as outputing sequences of lambda expressions, SQL queries, or programming languages. 
Our prediction is obtained through a large number of text-text pairs interpreted as squence-sequence pairs in a neural network or markov chainsetting.
In this paper we try to figure out how to use information in different domains in order to boos the performance of the running domain. 
Let's say you have many examples in publication domain but a few in education.
If I assume publications is partly related to education, use of it as training data seems almost necessary. 
But how much and which examples, this is the motivation of this paper. 


<table style="width:100%">
  <tr>
    <th>Firstname</th>
    <th>Lastname</th> 
    <th>Age</th>
  </tr>
  <tr>
    <td>Jill</td>
    <td>Smith</td> 
    <td>50</td>
  </tr>
  <tr>
    <td>Eve</td>
    <td>Jackson</td> 
    <td>94</td>
  </tr>
</table>


```
@ARTICLE{Javid:20018,
   author = {Dadashkarimi, J. and Fabbri, A. and Tatikonda, S. and 
  Radev, D.~R.},
    title = "{Zero-shot Transfer Learning for Semantic Parsing}",
  journal = {ArXiv e-prints},
archivePrefix = "arXiv",
   eprint = {1808.09889},
 primaryClass = "cs.CL",
 keywords = {Computer Science - Computation and Language, Computer Science - Machine Learning, Statistics - Machine Learning},
     year = 2018,
    month = aug,
   adsurl = {http://adsabs.harvard.edu/abs/2018arXiv180809889D},
  adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
