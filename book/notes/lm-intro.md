## Language Modeling

Language modeling is the task of predicting the next word in a sequence of words. It is a type of machine learning algorithm that is used to generate text.

The most common type of language model is called an autoregressive language model. It works by looking at a series of letters and trying to guess what letter comes next.

An autoregressive language model is called "autoregressive" because it predicts the next element (like a character or word) based on the previous elements, similar to how a variable in statistics might depend on its own past values.

  - "Auto" means "self" or "on its own"
  - "Regressive" refers to "going back" or "depending on previous values"
  
So "autoregressive" essentially means the model relies on its own previous outputs or values to make predictions. It's like the model is constantly looking back at what it has already seen or produced to figure out what comes next.

We're going to start by making a **character-level** language model. This is a kind of autoregressive model that looks at one letter at a time. We'll feed it a bunch of **names**, letter by letter. The model will try to learn patterns in these names. Once it learns these patterns, we can use it to create new names or predict what letter might come next in a name.

We will build different types of neural network models for this task, including:
  - Simple Bigram model (one character predicts the best or most likely next character)
  - Single linear layer of neural network Bigram Model
  - Language Model using MLP (multi-layer perceptron)
  - Transformer

We will use the **Character-Level Transformer Model** for our final model, as it is the most powerful and flexible model for this task.
