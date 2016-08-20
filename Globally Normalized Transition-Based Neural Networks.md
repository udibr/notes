# Globally Normalized Transition-Based Neural Networks
https://arxiv.org/abs/1603.06042

[Parsey McParseface](http://github.com/tensorflow/models/tree/master/syntaxnet) is  a parser of English sentences capable of finding parts of speech and dependency parsing. By Michael Collins and google NY.

This paper is more than just about google's data collection and computing powers. The parser uses a feed forward NN, which is much faster than the RNN usually used for parsing. Also the paper is using a global method to solve the label bias problem. This method can be used for many tasks and indeed in the paper it is used also to shorten sentences by throwing unnecessary words.

The label bias problem arises when predicting each label in a sequence using a softmax over all possible label values in each step. This is a local approach but what we are really interested in is a global approach in which the sequence of all labels that appeared in a training example are normalized by all possible sequences. This is intractable so instead a beam search is performed to generate alternative sequences to the training sequence. The search is stopped when the training sequence drops from the beam or ends. The different beams with the training sequence are then used to compute the global loss. 

Similar method is used in [seq2seq by Sasha Rush](https://github.com/udibr/notes/blob/master/Talk%20by%20Sasha%20Rush%20-%20Interpreting%2C%20Training%2C%20and%20Distilling%20Seq2Seq%E2%80%A6.pdf)