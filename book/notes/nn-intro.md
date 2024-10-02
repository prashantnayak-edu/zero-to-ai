## Introduction to Neural Networks

Neural Networks are just a specific class of mathematical expressions. The mathematical expression is designed to recognize patterns and relationships in data. 

They are inspired by the biological neural networks found in animal brains and are loosely based on the currently understood biology of neurons.

### Biological Motivation

![Biological Neuron](https://cs231n.github.io/assets/nn1/neuron.png)

The basic computational unit of the brain is a *neuron*. The brain contains approximately 86 billion neurons, each connected to others via *synapses*. The diagram above is a simplified (cartoon) illustration of a biological neuron.

The input structures of a neuron are called *dendrites*, and a neuron can have multiple dendrites that receive signals from other neurons. The output structure is called an *axon*, which can branch out and connect to the dendrites of many other neurons.

Our current understanding is that the nervous system delivers inputs to the brain in the form of electrical signals. These signals are received by the dendrites of neurons. If a neuron’s input signals reach a certain threshold, the neuron “fires,” sending an electrical impulse down its axon. This output signal is then transmitted to the dendrites of connected neurons, potentially causing them to fire as well. This chain reaction enables complex communication within the brain, forming the basis of all our thoughts, feelings, and actions.


### Mathematical Model 

Neural networks are artificial, mathematical representations of a network of neurons. The goal is to create this network and to *train* it to recognize patterns and relationships in data.

To better understand how neurons in neural networks work, let's start with a simple equation (the mathematical representation or expression) that you might already be familiar with from algebra.

#### Simple Linear Equation Example

In algebra, the equation of a straight line is given by:

$$
y = mx + c
$$

- y: The output or dependent variable.
- x: The input or independent variable.
- m: The slope of the line, representing how steep the line is. This can be thought of as a weight that scales the input.
- c: The y-intercept, which shifts the line up or down on the graph.

In this equation:

- The output y depends on the input x multiplied by the weight m, plus a constant c.
- Changing the value of m changes how much x influences y. A larger m means x has a bigger impact on y.

#### Extending to Multiple Inputs

If we have more than one input, the equation extends to:

$$
y = m_1 x_1 + m_2 x_2 + m_3 x_3 + \dots + c
$$

- $x_1, x_2, x_3, \dots$: Multiple input variables.
- $m_1, m_2, m_3, \dots$: Corresponding weights for each input.

Each input x_i is multiplied by its weight m_i, and all the products are added together along with the constant c to produce the output y.

#### Connecting to the Neuron model

![Mathematical model of a Neuron](https://cs231n.github.io/assets/nn1/neuron_model.jpeg)

- Inputs $x_i$: Represented by the arrows entering the neuron.
- Weights $w_i$: Associated with each input, indicating the strength of the connection.
- Summation $Σ$: The neuron sums all the weighted inputs.
- Activation Function $f$: Applied to the weighted sum to produce the output.

In the computational model of a neuron, we use a similar idea:

1. Inputs and Weights:
   - The neuron receives multiple inputs $(x_0, x_1, x_2, \dots)$, each representing a signal or piece of data.
   - Each input has an associated weight $(w_0, w_1, w_2, \dots)$ that determines its influence on the neuron's output.

2. Weighted Sum:
   - The neuron calculates a weighted sum of its inputs:
     $$
     Σ = w_0 x_0 + w_1 x_1 + w_2 x_2 + \dots + b
     $$
   - b is the bias term, similar to the constant c in the linear equation, which allows us to shift the activation function to the left or right.

3. Activation Function:
   - The neuron applies an activation function f to the weighted sum to determine the output:
     $$
     \text{Output} = f(Σ)
     $$
   - The activation function introduces non-linearity, enabling the network to learn complex patterns. 
   - In simpler terms, the typical activation function converts the output of the function (the weighted sum of inputs) into a value beween (-1, 1) or sometimes between (0,1).  The way to think about this is that a value of 0 or -1 means "off" (the neuron does not fire) and a value of 1 means "on" (the neuron fires)

#### Understanding Weights and Bias:

- Weights ($w_i$):
  - Determine how much each input influences the output.
  - Can be positive or negative:
    - Positive Weight: The input has an excitatory effect, increasing the output.
    - Negative Weight: The input has an inhibitory effect, decreasing the output.
- Bias (b):
  - Allows the neuron to adjust the output independently of the inputs.
  - Similar to the c in $y = mx + c$, shifting the activation function.

#### Activation Function:

- The activation function f decides whether the neuron should "fire" based on the weighted sum.
- A common choice is the sigmoid function (σ), which maps any real number into a value between 0 and 1:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

  where z is the weighted sum.
- This function is helpful because:
  - It introduces non-linearity, allowing the network to learn more complex patterns.
  - The output can be interpreted as a probability.

### Putting It All Together:

1. Step 1: Each input x_i is multiplied by its weight w_i.
2. Step 2: All the weighted inputs are summed together along with the bias b.
3. Step 3: The activation function f is applied to the sum to produce the neuron's output.
4. Step 4: The output can then be sent to other neurons in the network.

#### Example with Numbers:

Let's consider a simple neuron with two inputs:

- Inputs: $x_1 = 2$, $x_2 = 3$
- Weights: $w_1 = 0.5$, $w_2 = -1$
- Bias: $b = 1$

##### Calculate the Weighted Sum:

$$
\begin{align*}
\text{Weighted Sum} & = w_1 x_1 + w_2 x_2 + b \\
& = (0.5)(2) + (-1)(3) + 1 \\
& = 1 - 3 + 1 \\
& = -1
\end{align*}
$$

##### Apply Activation Function (Sigmoid):

$$
\begin{align*}
\text{Output} & = \sigma(-1) \\
& = \frac{1}{1 + e^{-(-1)}} \\
& = \frac{1}{1 + e^{1}} \\
& \approx 0.2689
\end{align*}
$$

#### Interpretation:

- The neuron's output is approximately 0.27.
- Since the output is closer to 0, the neuron is less likely to "fire" strongly.

#### Why This Matters:

- By adjusting the weights and bias, the neuron can learn to produce desired outputs for given inputs.
- In a network, multiple neurons work together, each learning different patterns.
- This simple mathematical model forms the basis for complex neural networks capable of tasks like image recognition, language translation, and more.

### Training the Neuron to Recognize Patterns

In our earlier example, we assumed specific values for the weights and bias to calculate the neuron's output. But how does a neuron determine the correct weights and bias in the first place? This is where training comes into play.

#### How Training Works:

1. Start with Initial Weights:
   - The neuron begins with random or initial guesses for the weights (w_i) and bias (b).
2. Provide Input and Desired Output:
   - We feed the neuron an input (e.g., $x_1 = 2, x_2 = 3$) and specify the desired output we want the neuron to produce.
3. Calculate the Neuron's Output:
   - Using the current weights and bias, the neuron computes its output following the steps we've discussed.
4. Compute the Error:
   - The neuron compares its calculated output to the desired output to find the error (the difference between them).
5. Adjust Weights and Bias:
   - The neuron adjusts its weights and bias to minimize the error. This adjustment is done in small increments.
6. Repeat the Process:
   - Steps 2–5 are repeated with multiple inputs and desired outputs. Over time, the neuron fine-tunes its weights and bias to improve its accuracy.

#### Relating to Our Example:

Let's see how this works using our previous numerical example.

##### Training Data:

- Input: $x_1 = 2$, $x_2 = 3$
- Desired Output: Let's say we want the neuron to output 1 for this input.

##### Initial Weights and Bias:

- Weights: $w_1 = 0.5$, $w_2 = -1$
- Bias: $b = 1$

##### Step 1: Calculate the Neuron's Output

Using the initial weights and bias:

$$
\begin{align*}
\text{Weighted Sum} & = w_1 x_1 + w_2 x_2 + b \\
& = (0.5)(2) + (-1)(3) + 1 \\
& = 1 - 3 + 1 \\
& = -1 \\
\\
\text{Output} & = \sigma(-1) \\
& = \frac{1}{1 + e^{1}} \\
& \approx 0.2689
\end{align*}
$$

##### Step 2: Compute the Error

$$
\text{Error} = \text{Desired Output} - \text{Actual Output} = 1 - 0.2689 = 0.7311
$$

##### Step 3: Adjust Weights and Bias

To reduce the error, we'll adjust the weights and bias slightly. The exact adjustment can involve some calculations, but we'll illustrate it simply here.

- Adjustments:
  - Increase $w_1$ slightly.
  - Increase $w_2$ slightly (since $w_2$ is negative, making it less negative increases it).
  - Adjust $b$ as needed.

Let's adjust the weights and bias:

- New Weights and Bias:
  - $w_1 = 0.7$
  - $w_2 = -0.8$
  - $b = 1.1$

##### Step 4: Recalculate the Output

$$
\begin{align*}
\text{Weighted Sum} & = (0.7)(2) + (-0.8)(3) + 1.1 \\
& = 1.4 - 2.4 + 1.1 \\
& = 0.1 \\
\\
\text{Output} & = \sigma(0.1) \\
& = \frac{1}{1 + e^{-0.1}} \\
& \approx 0.5249
\end{align*}
$$

##### Step 5: Compute the New Error

$$
\text{Error} = 1 - 0.5249 = 0.4751
$$

##### Result:

- The error decreased from 0.7311 to 0.4751.
- By adjusting the weights and bias, the neuron's output moved closer to the desired output.

#### Repeating the Process:

We continue this process:

- Adjust Weights and Bias Again:
  - Let's adjust to:
    - $w_1 = 0.9$
    - $w_2 = -0.6$
    - $b = 1.2$
- Recalculate Output:
  $$
  \begin{align*}
  \text{Weighted Sum} & = (0.9)(2) + (-0.6)(3) + 1.2 \\
  & = 1.8 - 1.8 + 1.2 \\
  & = 1.2 \\
  \\
  \text{Output} & = \sigma(1.2) \\
  & = \frac{1}{1 + e^{-1.2}} \\
  & \approx 0.7685
  \end{align*}
  $$
- Compute New Error:
  $$
  \text{Error} = 1 - 0.7685 = 0.2315
  $$
- The error has decreased further.

#### Understanding the Process:

- Goal: Reduce the error between the neuron's output and the desired output.
- Method: Adjust the weights and bias in small steps in the direction that reduces the error.
- Result: Over time, the neuron "learns" the correct weights and bias to produce outputs close to the desired outputs for given inputs.

#### Simplifying the Concept:

- Imagine trying to find the right combination of ingredients in a recipe to get the perfect taste.
- Each time you make the dish, you adjust the amounts slightly based on how close the taste is to what you want.
- Similarly, the neuron adjusts its weights and bias based on how close its output is to the desired output.

#### Why This Matters:

- By adjusting weights and biases through training, neurons in a network learn to recognize patterns in data.
- This learning allows the network to make accurate predictions or classifications when given new, unseen inputs.

### Key Takeaways 

- Neurons in Neural Networks:
  - Function similarly to equations you're familiar with, like y = mx + c.
  - Use weights to determine the influence of each input.
  - Apply activation functions to introduce non-linear behavior.

- Learning Process:
  - The network learns by adjusting the weights and biases based on data.
  - The goal is to minimize the difference between the predicted outputs and the actual outputs.

- Real-World Applications:
  - Neural networks are used in various technologies you interact with daily, such as smartphone voice assistants, facial recognition systems, and recommendation algorithms.

By relating neural networks to concepts you already understand, like linear equations, it's easier to grasp how they work and why they're powerful tools in modern technology.
