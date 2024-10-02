## Learning from data

In machine learning, we typically create models that predict or classify input values.

To get an intution about a model, imagine that it is like an equation.  

We are all familar with simple equations like 

$$
y = 2x + 3
$$

This represents a simple, linear relationship between x and y. For any value $x \in R$, it is quite easy to determine the value of y.

In other words, this model predicts the value of y, given any $x \in R$

In machine learning, we usually work with lots of data. For example, we might collect data on the square footage of homes and their selling prices. In this case, x is the square footage, and y is the price of the home.

Sometimes the relationship between x and y is more complex. For example, it might follow a curve like:

$$ y = wx^2 + b $$

This is a quadratic equation, which means y depends on the square of x.  (_Note: Remember that a quadratic equation is simply whan the relationship between $x$ and $y$ involes squared terms_)

In more complicated cases, we might use multiple inputs. For instance, we could also include the year the home was built. These multiple inputs are typically called _features_ in machine learning (think features of a home). The equation might look like:

$$ y = w_1x_1^2 + w_2x_2 + b $$

Here, $x_1$ represents the square footage, and $x_2$ represents the year built. 

In this case, the relationship between ($x_1$, $x_2$) and $y$ are a more complex, like plotting points on a 3D graph

These features are often put together into a feature vector:

$$ V = [x_1, x_2] $$

A feature vector is just a list of numbers representing the different signals or measurements we have.


### Formalizing the Idea

Lets put this together a bit more formally

Imagine we want to predict something, like the prices of a home based on a square footage of the home. The square footage is our signal $x$, and the prices of the home is our output $y$.

In many real-world situations, especially with high-dimensional data (like images), it’s hard to come up with a simple formula that directly relates $x$ to $y$.

Instead, we collect a large set of examples, called the training set. This training set contains pairs $(x_n, y_n)$, where $x_n$ is the input and $y_n$ is the _correct_ output.  So in our case, we collect many pairs of square footage and selling price for homes in our area.

We then create a model, which is a function $f$ with parameters $w$. These parameters are numbers that the model can adjust to improve its predictions.

For example, let’s say our function $f$ is predicting the price of a home based on its square footage:

$$ f(x; w) = w \cdot x $$

Here, $f(x; w)$ represents the predicted price of a home based on the input $x$  (the square footage). The parameter $w$  is the model’s weight, which determines how much the square footage influences the price prediction. The key point to note ther is that if we adjust $w$, the model changes its predictions.  As we will see later, when we speak about training the model, we are simply referring to finding the best value of $w$

If  $w = 300$ , then for a home of 1000 square feet, the predicted price would be:

$$ f(1000; 300) = 300 \cdot 1000 = 300,000 $$

So in this case, the model predicts that a 1000-square-foot home will sell for $300,000.

Training the model means finding the best values for w so that the model’s predictions are as close as possible to the actual outputs in the training set.

We measure how well the model is doing using a loss function, written as $L(w)$. The loss function is small when the model’s predictions are good, and our goal is to find w that makes the loss as small as possible.

For example, if we use the mean squared error (MSE) as the loss function, it works like this:

Let’s say we have a few data points of actual home prices and their square footage:

  - $x_1 = 1000$  sq ft, actual price  $y_1 = 300,000$
  - $x_2 = 2000$  sq ft, actual price  $y_2 = 500,000$

Our model predicts prices using  $f(x; w) = w \cdot x$ , and let’s say for now,  $w = 250$. Then, the model’s predictions would be:

  - Predicted price for  $x_1$  (1000 sq ft) is  $f(1000; 250) = 250 \cdot 1000 = 250,000$
  - Predicted price for  $x_2$  (2000 sq ft) is  $f(2000; 250) = 250 \cdot 2000 = 500,000$

Now, we can calculate the mean squared error (MSE) to see how far off the predictions are:

$$ MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - f(x_i; w))^2 $$

In this case:

$$ MSE = \frac{1}{2} \left[ (300,000 - 250,000)^2 + (500,000 - 500,000)^2 \right] $$

$$ MSE = \frac{1}{2} \left[ 50,000^2 + 0^2 \right] = \frac{1}{2} \left[ 2.5 \times 10^9 \right] = 1.25 \times 10^9 $$

So, the mean squared error here is 1.25 billion. If we adjust  w  to a better value, we can reduce this error and improve the model’s predictions.



