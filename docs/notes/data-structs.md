## Introduction to Linear Algebra: Data Structures

Linear algebra is an important branch of mathematics that deals with data structures like scalars, vectors, matrices, and sets. These are the building blocks for many operations in mathematics, physics, and even computer science. In this lesson, we will define each of these structures.

### Scalars

A **scalar** is just a single number. It could be any real number, such as:

$5$, $-2.5$, or $\pi$.

In linear algebra, scalars often represent magnitudes, quantities, or other single values.

### Vectors

A **vector** is an ordered list of numbers. It can be thought of as a point in space or an arrow with both magnitude and direction. For example, a vector in two dimensions (2D) might look like this:

$\mathbf{v} = \begin{bmatrix} 3 \\ 4 \end{bmatrix}$

This vector represents a point that is 3 units in the $x$ direction and 4 units in the $y$ direction. Vectors can have more dimensions, like 3D vectors or even higher.

### Matrices

A **matrix** is a grid of numbers arranged in rows and columns. For example, a 2x2 matrix looks like this:

$$
A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}
$$

Each element in the matrix is a number, and matrices are used to perform operations like transforming vectors or solving systems of equations.

### Sets

A **set** is a collection of distinct objects, often numbers, that do not have any particular order. For example, the set of even numbers between 1 and 10 is:

$$ S = \{ 2, 4, 6, 8, 10 \} $$

Sets are useful when we want to define a collection of elements with certain properties.


## Operations on Vectors

Vectors can be combined or manipulated using different operations. Here are some common ones:

### 1. Vector Addition (Sum)

To add two vectors, we add their corresponding components. For example, if we have two vectors:

$$ \mathbf{a} = \begin{bmatrix} 1 \\ 2 \end{bmatrix}, \quad \mathbf{b} = \begin{bmatrix} 3 \\ 4 \end{bmatrix} $$

Their sum is:

$$ \mathbf{a} + \mathbf{b} = \begin{bmatrix} 1 + 3 \\ 2 + 4 \end{bmatrix} = \begin{bmatrix} 4 \\ 6 \end{bmatrix} $$

### 2. Vector Subtraction (Difference)

To subtract two vectors, we subtract their corresponding components. For example:

$$ \mathbf{a} - \mathbf{b} = \begin{bmatrix} 1 - 3 \\ 2 - 4 \end{bmatrix} = \begin{bmatrix} -2 \\ -2 \end{bmatrix} $$

### 3. Scalar Multiplication

When multiplying a vector by a scalar (a single number), we multiply each component of the vector by that scalar. For example, if we multiply the vector $\mathbf{a}$ by 2:

$$ 2 \cdot \mathbf{a} = 2 \cdot \begin{bmatrix} 1 \\ 2 \end{bmatrix} = \begin{bmatrix} 2 \\ 4 \end{bmatrix} $$

### 4. Dot Product

The dot product of two vectors is a single number that comes from multiplying corresponding components of two vectors and then adding them. For example:

$$ \mathbf{a} \cdot \mathbf{b} = (1 \cdot 3) + (2 \cdot 4) = 3 + 8 = 11 $$

The dot product is used to find the angle between two vectors and to determine if they are orthogonal (perpendicular).

## Intuition on Vectors: Hiking a Hill

To build a better understanding of vectors, let's use the analogy of hiking. Imagine you are hiking up a hill, and you want to reach the top as efficiently as possible.

### Hiking in 2D

In two dimensions, imagine you're on a hill with a trail going uphill and downhill. The slope of the hill can be represented by a vector. This vector tells you which direction is the steepest and how steep the hill is.

If you're trying to reach the top of the hill, you'd want to follow the direction that points the steepest upwards—this is your gradient vector, which we will cover more in the future lessons on derivatives and functions.

### Hiking in 3D

Now imagine you're hiking on a 3D mountain. Again, a vector represents the direction and steepness of the terrain in all directions. If you're at a point on the mountain and want to move uphill, the vector gives you both the direction you should hike and how steep the climb will be.

By understanding how vectors work, you can visualize moving through space or solving real-world problems where both direction and magnitude matter.

This intuition will be important as we continue to explore more advanced topics in linear algebra, such as gradients and minimizing functions. The key takeaway here is that vectors help us describe movement, direction, and how quantities change in space.
