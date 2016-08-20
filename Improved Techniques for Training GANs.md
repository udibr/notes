# Improved Techniques for Training GANs 
http://arxiv.org/abs/1606.03498


[code](https://github.com/openai/improved-gan),
[demo](http://infinite-chamber-35121.herokuapp.com/cifar-minibatch/1/?),
[related](http://www.inference.vc/understanding-minibatch-discrimination-in-gans/)

### Feature matching
problem: overtraining on the current discriminator

solution:
￼$||E_{x \sim p_{\text{data}}}f(x) - E_{z \sim p_{z}(z)}f(G(z))||_{2}^{2}$

were f(x) activations intermediate layer in discriminator
### Minibatch discrimination
problem: generator to collapse to a single point

solution: for each sample i, concatenate to $f(x_i)$ features $b$ measuring its distance to other samples j (i and j are both real or generated samples in same batch):
$\sum_j \exp(-||M_{i, b} - M_{j, b}||_{L_1})$
￼
this generates visually appealing samples very quickly
### Historical averaging
problem: SGD fails by going into extended orbits

solution: parameters revert to the mean
$|| \theta - \frac{1}{t} \sum_{i=1}^t \theta[i] ||^2$
￼
### One-sided label smoothing
problem: discriminator vulnerability to adversarial examples

solution: discriminator target for positive samples is 0.9 instead of 1

### Virtual batch normalization
problem: using BN cause output of examples in batch to be dependent

solution: use reference batch chosen once at start of training and each sample is normalized using itself and the reference. It's
expensive so used only on generation

### Assessment of image quality
problem: MTurk not reliable

solution: use inception model p(y|x) to compute 
￼$\exp(\mathbb{E}_x \text{KL}(p(y | x) || p(y)))$
on 50K generated images x

### Semi-supervised learning
use the discriminator to also classify on K labels when known and
use all real samples (labels and unlabeled) in the discrimination task
￼$D(x) = \frac{Z(x)}{Z(x) + 1}, \text{ where } Z(x) = \sum_{k=1}^{K} \exp[l_k(x)]$.
In this case use feature matching but not minibatch discrimination.
It also improves the quality of generated images.
