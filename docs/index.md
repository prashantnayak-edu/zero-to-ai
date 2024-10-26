# Zero to AI
##### *- Prashant Nayak*
---

This course provides a comprehensive journey through the foundations of modern AI, covering topics from basic math and machine learning to advanced neural networks and reinforcement learning. It's designed to connect the dots in a way that many traditional AI courses don't.

While the course covers a wide range of topics, special emphasis is placed on Neural Networks and the Transformer architecture, which are at the heart of the current AI revolution. These technologies are driving the most significant breakthroughs in AI today.

I initially created this course to explain AI to my undergraduate daughter studying Math. As I developed it, I realized its potential to reach even younger students, like my soon-to-be 11th grade daughter. My goal evolved into crafting explanations that a motivated and intelligent high school student could grasp.

The importance of this material can't be overstated. AI is shaping up to be the most significant technological shift since the World Wide Web emerged in the 1990s - potentially 100 times more impactful. Understanding these concepts is becoming crucial for anyone looking to navigate our rapidly changing world.

This course aims to demystify AI, breaking down complex ideas into digestible pieces. Whether you're a student, a professional, or simply curious about AI, I hope this material will help you understand the technology that's reshaping our future.

### Sources

We stand on the shoulders of giants in creating this course, drawing inspiration and knowledge from the brilliant work of the following educators and researchers in the field of AI and machine learning whose (open source) material I have used and is sourced from:
- [Andrej Karpathy - NN-Zero-To-Hero](https://github.com/karpathy/nn-zero-to-hero)
- [Andriy Burkov - The Hundred-Page Machine Learning Book](http://themlbook.com)
- [CS-231 Stanford University](https://cs231n.github.io/)
- [Vik Parchuri - Zero_To_GPT](https://github.com/VikParuchuri/zero_to_gpt/tree/master?tab=readme-ov-file)
- [Tivadar Danka - Mathematics of Machine Learning](https://tivadardanka.com/mathematics-of-machine-learning-preview)

## Course Outline

### Part 1 - Foundations

1. **Math Fundamentals for Machine Learning and AI**
   - [Terminology and Notation](./notes/term-not.html)
   - [Data Structures - Scalars, Vectors, Matrices](./notes/data-structs.html)
   - [Functions, Derivatives and Gradients](./notes/func-der-grad.html)
   - [Derivative of a function of a single variable](./notes/derivative-single-var.html)
   - [Derivative of a function of multiple variables](./notes/derivative-multiple-var.html)

2. **Introduction to Machine Learning**
   - [What is Machine Learning](./notes/what-is-ml.html)
   - [Prediction and Classification](./notes/prediction-classification.html)
   - [Why and how does an algorithm "learn"?](./notes/why-algo-learns.html)
   - [Difference between Machine Learning and AI](./notes/ml-ai.html)

3. **Learning from data**
   - [Learning from data](./notes/learning-from-data.html)
   - [Fundamental Algorithms for Prediction - Linear and Logistic Regression](./notes/fundamental-algorithms.html)
       - introduces Loss functions like Mean Squared Error (MSE) and Cross Entropy Loss
       - introduces Optimization algorithms like Gradient Descent with a single input
   - [Optimization Algorithms - Gradient Descent and Stochastic Gradient Descent](./notes/gradient-descent.html)
   - [Fundamental Algorithm for Classification - K-Nearest Neighbors](./notes/knn.html)

4. **Introduction to Feature Engineering**
   - [What is Feature Engineering](./notes/feature-engineering.html)
   - [One-hot encoding](./notes/one-hot-encoding.html)
   - [Normalization and Standardization](./notes/normalization-standardization.html)
   - [Dealing with missing features and Data Imputation Techniques](./notes/missing-values-imputation.html)

### Part 2 - Neural Networks, Language Models and Transformers

5. **Neural Networks**
   - [Introduction to Neural Networks](./notes/nn-intro.html)
   - [Types of Neural Networks](./notes/nn-types.html)
   - [Neural Networks Part 1 - Forward Pass & Backpropagation (MicroGrad)](./notes/nn-forward-backprop.html)
   - [Neural Networks Part 2 - Training (MicroGrad)](./notes/nn-training.html)
   - [Neural Networks Part 3 - PyTorch](./notes/nn-pytorch.html)

6. **Optimization and Regularization**
   - [Softmax and Cross-Entropy](./notes/softmax-cross-entropy.html)
   - [Regularization](./notes/regularization.html)

7. **Vector Embeddings**
   - [GPT style tokenization](./notes/gpt-tokenization.html)
   - [Vector Embeddings](./notes/vector-embeddings.html)
   - [Vector Store (MicroVectorStore)](./notes/vector-store.html)

8. **Working with Text (basics of GPT)**
   - [Language Modeling Introduction](./notes/lm-intro.html)
   - [Simple Bigram model](./notes/bigram-lm.html)
   - [Single linear layer of neural network Bigram Model](./notes/bigram-nn-lm.html)
   - [Language Model using MLP (multi-layer perceptron)](./notes/bigram-mlp-lm.html)
   - [Building GPT-2 - Transformer Language Model](./notes/bigram-transformer-lm.html)

### Part 3 - Advanced Topics

9. **More neural network types**
   - [Convolutional Neural Networks](./notes/conv-nn.html)
   - [Recurrent Neural Networks](./notes/rnn.html)
   - [Classification with Neural Networks](./notes/nn-classification.html)

10. **Introduction to Reinforcement Learning**
    - [Introduction to Reinforcement Learning](./notes/rl-intro.html)
    - [Markov Decision Processes](./notes/mdp.html)
    - [Reinforcement Learning - Q-learning](./notes/q-learning.html)
    - [Reinforcement Learning - Policy Gradients](./notes/policy-gradients.html)
    - [Reinforcement Learning - Actor-Critic](./notes/actor-critic.html)
    - [Reinforcement Learning - Deep Reinforcement Learning](./notes/deep-rl.html)

### Part 4 - Additional ML and Data Science Topics
   - [75 terms every data scientist should know](./notes/75-terms-ds.html)
   - [Modular AI Programs using DSPy](./notes/modular-ai-programs.html)


### Miscellaneous
- [C++ for Java Students](./notes/cplusplusforjava.html)


