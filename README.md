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
- [Vik Parchuri - Zero_To_GPT](https://github.com/VikParuchuri/zero_to_gpt/tree/master?tab=readme-ov-file)
- [Tivadar Danka - Mathematics of Machine Learning](https://tivadardanka.com/mathematics-of-machine-learning-preview)
- [CS-231 Stanford University](https://cs231n.github.io/)

## Course Outline

### Part 1 - Foundations

1. **Math Fundamentals**: Lessons on basics of linear algebra and calculus
   - [Terminology and Notation](./notes/term-not.ipynb)
   - [Data Structures - Scalars, Vectors, Matrices](./notes/data-structs.ipynb)
   - [Introduction to Functions, Derivatives and Gradients](./notes/func-der-grad.ipynb)
   - [Derivative of a function of a single variable](./notes/derivative-single-var.ipynb)
   - [Derivative of a function of multiple variables](./notes/derivative-multiple-var.ipynb)

2. **Machine Learning Basics**
   - [What is Machine Learning and types](./notes/what-is-ml.ipynb)
   - [Prediction and Classification](./notes/prediction-classification.ipynb)
   - [Why and how does an algorithm "learn"](./notes/why-algo-learns.ipynb)

3. **Learning from data**
   - [Learning from data](./notes/learning-from-data.ipynb)
   - [Fundamental Algorithms for prediction - Linear and Logistic Regression](./notes/fundamental-algorithms.ipynb)
       - also introduces Loss functions like Mean Squared Error (MSE) and Cross Entropy Loss
       - also introduces Gradient Descent with a single input
   - [Optimization Algorithms - Gradient Descent and Stochastic Gradient Descent](./notes/gradient-descent.ipynb)
   - [Fundamental Algorithm for classificaiton - K-Nearest Neighbors](./notes/knn.ipynb)

4. **Feature Engineering**
   - [Introduction to Feature Engineering](./notes/feature-engineering.ipynb)
   - [One-hot encoding](./notes/one-hot-encoding.ipynb)
   - [Normalization and Standardization](./notes/normalization-standardization.ipynb)
   - [Dealing with missing features and Data Imputation Techniques](./notes/missing-values-imputation.ipynb)

### Part 2 - Neural Networks, Language Models and Transformers

5. **Neural Networks**
   - [Introduction to Neural Networks](./notes/nn-intro.ipynb)
   - [Types of Neural Networks](./notes/nn-types.ipynb)
   - [Neural Networks Part 1 - Forward Pass & Backpropagation (MicroGrad)](./notes/nn-forward-backprop.ipynb)
   - [Neural Networks Part 2 - Training (MicroGrad)](./notes/nn-training.ipynb)
   - [Neural Networks Part 3 - PyTorch](./notes/nn-pytorch.ipynb)

6. **Optimization and Regularization**
   - [Softmax and Cross-Entropy](./notes/softmax-cross-entropy.ipynb)
   - [Regularization](./notes/regularization.ipynb)

7. **Vector Embeddings**
   - [GPT style tokenization](./notes/gpt-tokenization.ipynb)
   - [Vector Embeddings](./notes/vector-embeddings.ipynb)

8. **Working with Text (basics of GPT)**
   - [Language Modeling Introduction](./notes/lm-intro.ipynb)
   - [Simple Bigram model](./notes/bigram-lm.ipynb)
   - [Single linear layer of neural network Bigram Model](./notes/bigram-nn-lm.ipynb)
   - [Language Model using MLP (multi-layer perceptron)](./notes/bigram-mlp-lm.ipynb)
   - [Building GPT-2 - Transformer Language Model](./notes/bigram-transformer-lm.ipynb)

### Part 3 - Advanced Topics

9. **More neural network types**
   - [Convolutional Neural Networks](./notes/conv-nn.ipynb)
   - [Recurrent Neural Networks](./notes/rnn.ipynb)
   - [Classification with Neural Networks](./notes/nn-classification.ipynb)

10. **Introduction to Reinforcement Learning**
    - [Introduction to Reinforcement Learning](./notes/rl-intro.ipynb)
    - [Markov Decision Processes](./notes/mdp.ipynb)
    - [Reinforcement Learning - Q-learning](./notes/q-learning.ipynb)
    - [Reinforcement Learning - Policy Gradients](./notes/policy-gradients.ipynb)
    - [Reinforcement Learning - Actor-Critic](./notes/actor-critic.ipynb)
    - [Reinforcement Learning - Deep Reinforcement Learning](./notes/deep-rl.ipynb)

