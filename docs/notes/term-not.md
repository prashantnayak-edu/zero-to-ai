
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML" async></script>
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    tex2jax: {
        inlineMath: [['$','$'], ['\(','\)']],
        processEscapes: true
    }
});
</script>
## Lesson: Basic Notation in Mathematics

In this lesson, we'll cover some basic mathematical notations that are frequently used, especially in the context of summation, products, and finding maximum or minimum values.

### Capital Sigma (Summation)

The capital sigma \( $\sum$ \) is used to denote summation, which means adding up a series of numbers.

#### Example:

The summation of numbers from \( i = 1 \) to \( n \) is written as:

$$ \sum_{i=1}^{n} a_i $$

This means:

$$ a_1 + a_2 + a_3 + \dots + a_n $$

For example, if \( $a_i = i$ \), then:

$$ \sum_{i=1}^{4} i = 1 + 2 + 3 + 4 = 10 $$

### Capital Pi (Product)

The capital pi \( $\prod$ \) is used to denote the product of a series of numbers, which means multiplying a sequence of terms.

#### Example:

The product of numbers from \( i = 1 \) to \( n \) is written as:

$$ \prod_{i=1}^{n} a_i $$

This means:

$$ a_1*a_2 * a_3* \dots *a_n $$

For example, if \( $a_i = i$ \), then:

$$ \prod_{i=1}^{4} i = 1 * 2 * 3 * 4 = 24 $$

### Maximum (max) and Minimum (min)

The notation \( $\max$ \) and \( $\min$ \) are used to find the maximum or minimum value in a set of numbers.

#### Example:

If we have a set of numbers:

$$ [3, 5, 7, 2, 9] $$

Then:

$$ \max(3, 5, 7, 2, 9) = 9 $$

And:

$$ \min(3, 5, 7, 2, 9) = 2 $$

### Arg Max and Arg Min

The **arg max** and **arg min** functions return the index (or argument) at which the maximum or minimum value occurs.

#### Example:

For the same set of numbers \( $[3, 5, 7, 2, 9]$ \), the arg max and arg min are:

$$ 	argmax(3, 5, 7, 2, 9) = 5 $$ 

(since the maximum value 9 occurs at the 5th position).

$$ 	argmin(3, 5, 7, 2, 9) = 4 $$ 

(since the minimum value 2 occurs at the 4th position).

These notations are very useful in optimization problems, where you want to find the input that either maximizes or minimizes a certain function.

That's a brief overview of some of the basic mathematical notations used frequently. These concepts are especially important in linear algebra, calculus, and data science.

