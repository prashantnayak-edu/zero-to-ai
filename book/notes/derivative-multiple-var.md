## Derivative of a Function with Two Variables

In this lesson, we will extend the concept of derivatives to functions with two variables. The derivative of a function with multiple variables is called a **partial derivative**. Instead of finding how a function changes with respect to just one variable, we now see how the function changes with respect to **each** variable individually, while keeping the others constant.

### Definition of Partial Derivatives

If we have a function \( f(x, y) \), the **partial derivative** of \( f \) with respect to \( x \) (denoted 

$$
\frac{\partial f}{\partial x}
$$

measures how \( f \) changes as we vary \( x \) while keeping \( y \) constant. 

Similarly, the partial derivative with respect to \( y \) (denoted 

$$
\frac{\partial f}{\partial y}
$$

measures how \( f \) changes as we vary \( y \) while keeping \( x \) constant.

#### Example: 

Let's take the function:

$$ f(x, y) = x^2 + 3xy + y^2 $$

##### Finding the Partial Derivatives:

1. To find the partial derivative with respect to \( x \), treat \( y \) as a constant and differentiate with respect to \( x \):

$$ \frac{\partial f}{\partial x} = 2x + 3y $$

2. To find the partial derivative with respect to \( y \), treat \( x \) as a constant and differentiate with respect to \( y \):

$$ \frac{\partial f}{\partial y} = 3x + 2y $$

These partial derivatives tell us how \( f(x, y) \) changes with respect to changes in \( x \) and \( y \) individually.

##### Python Code for Partial Derivatives

We can use Python to compute the partial derivatives of a function. Below is some Python code to calculate the partial derivatives of the function \( f(x, y) = x^2 + 3xy + y^2 \) at a specific point.


```python
# Define the function
def f(x, y):
    return x**2 + 3*x*y + y**2

# Define the finite difference method to approximate partial derivatives
def partial_derivative_x(f, x, y, h=1e-5):
    return (f(x + h, y) - f(x, y)) / h

def partial_derivative_y(f, x, y, h=1e-5):
    return (f(x, y + h) - f(x, y)) / h

# Example point (x, y)
x_val = 2
y_val = 3

# Calculate partial derivatives at (2, 3)
df_dx = partial_derivative_x(f, x_val, y_val)
df_dy = partial_derivative_y(f, x_val, y_val)

df_dx, df_dy
```




    (13.000010000041582, 12.00001000007944)



### Explanation of Code:

- The function \( f(x, y) \) represents our mathematical function \( f(x, y) = x^2 + 3xy + y^2 \).
- The functions `partial_derivative_x` and `partial_derivative_y` use finite differences to approximate the partial derivatives with respect to \( x \) and \( y \).
- We evaluate the partial derivatives at \( x = 2 \) and \( y = 3 \).

As we learnt in the lesson [Introduction to Functions, Derivatives and Gradients](/notes/func-der-grad.ipynb), the vector of partial derivatives is called the *gradient*. In our case, it is like a map that tells us that at the point (2,3) the function is increasing the fastest in the direction of the $df/dx$ with a value of 13.


### Intuition on Partial Derivatives: Hiking on a 3D Terrain

Imagine you are hiking on a 3D terrain where the height of the terrain at each point is given by the function \( f(x, y) \). The two directions you can move in are along the \( x \)-axis (say, east-west) and the \( y \)-axis (say, north-south).

- **Partial Derivative with Respect to \( x \)**: If you only walk east-west (changing \( x \) while keeping \( y \) constant), the partial derivative \( \frac{\partial f}{\partial x} \) tells you how the height (function value) changes as you move in that direction.
- **Partial Derivative with Respect to \( y \)**: If you only walk north-south (changing \( y \) while keeping \( x \) constant), the partial derivative \( \frac{\partial f}{\partial y} \) tells you how the height changes as you move in that direction.

Together, these partial derivatives provide insight into how the terrain changes in different directions, helping you decide which way to move to either climb uphill or find a flat area (where both partial derivatives are zero).


### Summary:

- A **partial derivative** is the derivative of a multivariable function with respect to one variable, keeping the others constant.
- For a function \( f(x, y) \), we compute \( \frac{\partial f}{\partial x} \) and \( \frac{\partial f}{\partial y} \) to see how the function changes in the \( x \)- and \( y \)-directions, respectively.
- In the context of hiking on a 3D terrain, partial derivatives tell us the slope of the hill in different directions.