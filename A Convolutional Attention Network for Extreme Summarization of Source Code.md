# A Convolutional Attention Network for Extreme Summarization of Source Code
http://arxiv.org/abs/1602.03001

[web site](http://groups.inf.ed.ac.uk/cup/codeattention/), [code (Theano)](https://github.com/mast-group/convolutional-attention), [working version of code](https://github.com/udibr/convolutional-attention), [ICML](http://icml.cc/2016/?page_id=1839#971), [external notes](https://github.com/jxieeducation/DIY-Data-Science/blob/master/papernotes/2016/02/conv-attention-network-source-code-summarization.md)

Given an arbitrary snippet of Java code (~72 tokens) generate the methods name (~3 tokens):
generation starts with a $m_0 = \text{start-symbol}$ and state $h_0$, to generate next output token $m_t$ do:
* convert code tokens $c_i$ and embed to $E_{c_i}$
* convert all $E_{c_i}$ to $\alpha$ and $\kappa$ all same length as code using a network of `Conv1D` and padding (`Conv1D` because the code is highly structured, unambiguous.) The convertion is done using following network:
![](http://i.imgur.com/cHbiSIi.png?1)
* $\alpha$ and $\kappa$ are probabilities over length of code (using softmax).
* In addition compute $\lambda$ by running another `Conv1D` over $L_\text{feat}$ with $\sigma$ activation and take the maximal value. 
* use $\alpha$ to weight average $E_{c_i}$ and pass the average through FC layer to end with a softmax over output vocabulary $V$. Probability for output word $m_t$ is $n_{m_t}$.
* As an alternative use $\kappa$ to give probability to use as output each of the tokens $c_i$ which can be inside $V$ or outside it. This is also called "translation-invariant features" ([ref](https://papers.nips.cc/paper/5866-pointer-networks.pdf))
* $\lambda$ is used as a meta-attention deciding which to use:
$P(m_t \mid h_{t-1},c) = \lambda \sum_i \kappa_i I_{c_i = m_t} + (1-\lambda) \mu n_{m_t}$
where $\mu$ is $1$ unless you are in training and $m_t$ is UNK and the correct value for $m_t$ appears in $c$ in which case it is $e^{-10}$
* Advance $h_{t-1}$ to $h_t$ with GRU and using as input the embedding of output token $m_{t-1}$ (while training this is taken from the training labels or with small probability the argmax of the generated output.)
* Generating using hybrid breadth-first search and beam search: keep a heap of all suggestions and always try to extend the best suggestion so far. Remove suggestions that are worse than all the completed suggestions (dead) so far.


