
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML" async></script>
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    tex2jax: {
        inlineMath: [['$','$'], ['\(','\)']],
        processEscapes: true
    }
});
</script>
# Zero to AI

This course provides a comprehensive journey through the foundations of modern AI, covering topics from basic math and machine learning to advanced neural networks and reinforcement learning. It's designed to connect the dots in a way that many traditional AI courses don't.

While the course covers a wide range of topics, special emphasis is placed on Neural Networks and the Transformer architecture, which are at the heart of the current AI revolution. These technologies are driving the most significant breakthroughs in AI today.

I initially created this course to explain AI to my undergraduate daughter studying Math. As I developed it, I realized its potential to reach even younger students, like my soon-to-be 11th grade daughter. My goal evolved into crafting explanations that a motivated and intelligent high school student could grasp.

The importance of this material can't be overstated. AI is shaping up to be the most significant technological shift since the World Wide Web emerged in the 1990s - potentially 100 times more impactful. Understanding these concepts is becoming crucial for anyone looking to navigate our rapidly changing world.

This course aims to demystify AI, breaking down complex ideas into digestible pieces. Whether you're a student, a professional, or simply curious about AI, I hope this material will help you understand the technology that's reshaping our future.

## Sources

The material here is sourced from:
- [Andrej Karpathy - NN-Zero-To-Hero](https://github.com/karpathy/nn-zero-to-hero)
- [Vik Parchuri - Zero_To_GPT](https://github.com/VikParuchuri/zero_to_gpt/tree/master?tab=readme-ov-file)
- [Tivadar Danka - Mathematics of Machine Learning](https://tivadardanka.com/mathematics-of-machine-learning-preview)
- [CS-231 Stanford University](https://cs231n.github.io/)

## Course Outline

### Part 1 - Foundations

1. **Math Fundamentals**: Lessons on basics of linear algebra and calculus
   - [Terminology and Notation](./notes/term-not.md)
   - [Data Structures - Scalars, Vectors, Matrices](./notes/data-structs.md)
   - [Introduction to Functions, Derivatives and Gradients](./notes/func-der-grad.md)
   - [Derivative of a function of a single variable](./notes/derivative-single-var.md)
   - [Derivative of a function of multiple variables](./notes/derivative-multiple-var.md)

2. **Machine Learning**
   - [What is Machine Learning and types](./notes/what-is-ml.md)
   - [Why and how does an algorithm "learn"](./notes/why-algo-learns.md)
   - [Fundamental Algorithm - Linear Regression](./notes/linear-reg.md)
   - [Fundamental Algorithm - Logistic Regression](./notes/logistic-reg.md)

3. **Learning from data**
   - [Learning from data](./notes/learning-from-data.md)
   - [K-Nearest Neighbors](./notes/knn.md)
   - [Gradient Descent](./notes/gradient-descent.md)
   - [Stochastic Gradient Descent](./notes/sgd.md)

4. **Feature Engineering**
   - [Introduction to Feature Engineering](./notes/feature-engineering.md)
   - [One-hot encoding](./notes/one-hot-encoding.md)
   - [Normalization and Standardization](./notes/normalization-standardization.md)
   - [Dealing with missing features and Data Imputation Techniques](./notes/missing-values-imputation.md)

### Part 2 - Neural Networks, Language Models and Transformers

5. **Neural Networks**
   - [Introduction to Neural Networks](./notes/nn-intro.md)
   - [Types of Neural Networks](./notes/nn-types.md)
   - [Neural Networks Part 1 - Forward Pass & Backpropagation (MicroGrad)](./notes/nn-forward-backprop.md)
   - [Neural Networks Part 2 - Training (MicroGrad)](./notes/nn-training.md)
   - [Neural Networks Part 3 - PyTorch](./notes/nn-pytorch.md)

6. **Optimization and Regularization**
   - [Softmax and Cross-Entropy](./notes/softmax-cross-entropy.md)
   - [Regularization](./notes/regularization.md)

7. **Vector Embeddings**
   - [GPT style tokenization](./notes/gpt-tokenization.md)
   - [Vector Embeddings](./notes/vector-embeddings.md)

8. **Working with Text (basics of GPT)**
   - [Language Modeling Introduction](./notes/lm-intro.md)
   - [Simple Bigram model](./notes/bigram-lm.md)
   - [Single linear layer of neural network Bigram Model](./notes/bigram-nn-lm.md)
   - [Language Model using MLP (multi-layer perceptron)](./notes/bigram-mlp-lm.md)
   - [Building GPT-2 - Transformer Language Model](./notes/bigram-transformer-lm.md)

### Part 3 - Advanced Topics

9. **More neural network types**
   - [Convolutional Neural Networks](./notes/conv-nn.md)
   - [Recurrent Neural Networks](./notes/rnn.md)
   - [Classification with Neural Networks](./notes/nn-classification.md)

10. **Introduction to Reinforcement Learning**
    - [Introduction to Reinforcement Learning](./notes/rl-intro.md)
    - [Markov Decision Processes](./notes/mdp.md)
    - [Reinforcement Learning - Q-learning](./notes/q-learning.md)
    - [Reinforcement Learning - Policy Gradients](./notes/policy-gradients.md)
    - [Reinforcement Learning - Actor-Critic](./notes/actor-critic.md)
    - [Reinforcement Learning - Deep Reinforcement Learning](./notes/deep-rl.md)

