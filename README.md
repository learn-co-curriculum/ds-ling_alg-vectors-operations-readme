
## Vectors and Operations

* Describe a vector as a combination of scalar values along with geometric intuition. 
* Define a vector in python and using indexing to address/modify its elements.
* Apply basic vector-vector and vector-scalar arithmetic operations.
* Calculate dot products as a measure of similarity.

Let's get started with an introduction to vectors and matrices for machine learning. 

### What is a Vector?

A vector is a tuple of one or more values - known as scalar components of the vector.

> **Vectors are built from individual components, which are numerical in nature. We can think of a vector as a list of numbers (like the cost vector with `b` and `b` above), and vector algebra as operations performed on the numbers in the list**

Vectors are usually represented using a vertical or a horizontal orientation (analogous to column vs. row in a data table). This can be shown as :

$$
v = 
\left(\begin{array}{cc} 
v1 & v2 & v3 & v4\\
\end{array}\right)
$$ 

OR 

$$
v = 
\left(\begin{array}{cc} 
v1\\
v1\\
v1\\
v1\\
\end{array}\right)
$$ 

where `v` is the name of the vector and `v1,v2,v3,v4` are the scalar components of the vector. 

In machine learning systems, the output variable (e.g. the cost vector above) is known as a target vector with the lowercase `y` when describing the training of a machine learning algorithm.

#### A geometric intuition
A vector can be thought as an entity that represents spatial coordinates in an n-dimensional space, where n is the number of dimensions. A vector can also represent a line from the origin of the vector space with a direction and a magnitude, based on scalar components.

Let’s look at how to define a vector in Python.

### Defining a Vector in Python
In python, one of the easiest ways to represent a vector is using `Numpy` arrays. The list scalar values can be used to create a vector in python as shown below:


```python
# create a vector from list [2,4,6]
import numpy as np
v = np.array([2, 4, 6])
print(v)
```

    [2 4 6]


So now we have a vector with given values, we shall see later that we can rotate this to form a vertical vector using numpy's `.T` or `np.transpose()`, i.e. transpose the vector. 


```python
v.T
```




    array([2, 4, 6])



The commas between the elements after transpose signifies a new row. 

### Indexing a Vector

There are times when we have a lot of data in a vector (or array as we now know it) and we want to extract a portion of the data for some analysis. For example, maybe you want to know the first few values of a long array, or you want the integral of data between x = 4 and x = 6, but your vector covers 0 < x < 10. Indexing is the way to do these things. Let's generate a long vector to see this action using 10 values between -pi and pi (i.e. -3.14 to 3.14):


```python
x = np.linspace(-np.pi, np.pi, 10)
print(x)
```

    [-3.14159265 -2.44346095 -1.74532925 -1.04719755 -0.34906585  0.34906585
      1.04719755  1.74532925  2.44346095  3.14159265]


We can use the index values to address individual scalar values within this vector , similar to python list indexing as shown below:


```python
print (x[0])  # first element
print (x[2])  # third element
print (x[-1]) # last element
print (x[-2]) # second to last element
```

    -3.141592653589793
    -1.7453292519943295
    3.141592653589793
    2.443460952792061


We can select a range of elements too. The syntax `a:b` extracts the ath to b-1th elements. The syntax a:b:n starts at a, skips n elements up to the index b.


```python
print (x[1:4])     # second to fourth element. Element 5 is not included
print (x[0:-1:2])  # every other element
print (x[:])       # print the whole vector
print (x[-1:0:-1]) # reverse the vector!
```

    [-2.44346095 -1.74532925 -1.04719755]
    [-3.14159265 -1.74532925 -0.34906585  1.04719755  2.44346095]
    [-3.14159265 -2.44346095 -1.74532925 -1.04719755 -0.34906585  0.34906585
      1.04719755  1.74532925  2.44346095  3.14159265]
    [ 3.14159265  2.44346095  1.74532925  1.04719755  0.34906585 -0.34906585
     -1.04719755 -1.74532925 -2.44346095]


### Basic Vector Arithmetic 

Two vectors of **same length** can be added, subtracted, multiplied and divided - to create a new vector. For vector-to-vector arithmetic, we can use Numpy's built in basic arithmetic operations including addition, multiplication, subtraction and division as shown below:

#### Addition
Each element of the new vector is calculated by adding  elements of the other vectors at the same location. 

> **a + b = [ a1 + b1 , a2 + b2 , a3 + b3 ]**

Let's see this in action: 


```python
# create two arrays in numpy and add them - print results
v1 = np.array([2,4,6])
v2 = np.array([1,3,5])
v_added = v1 + v2
print (v1, '+', v2, '=', v_added)
```

    [2 4 6] + [1 3 5] = [ 3  7 11]


#### Subtraction

Each element of the new vector is calculated by subtracting elements of the other vectors at the same location. 

> **a - b = [ a1 - b1 , a2 - b2 , a3 - b3 ]**


```python
# create two arrays in numpy and subtract them - print results
v1 = np.array([2,4,6])
v2 = np.array([1,3,5])
v = v1 - v2
print (v1, '-', v2, '=', v)
```

    [2 4 6] - [1 3 5] = [1 1 1]


#### Multiplication
Each element of the new vector is calculated by multiplying elements of the other vectors at the same location.

> **a \* b = [ a1 \* b1 , a2 \* b2 , a3 \* b3 ]**


```python
# create two arrays in numpy and multiply them - print results
v1 = np.array([2,4,6])
v2 = np.array([1,3,5])
v = v1 * v2
print (v1, '*', v2, '=', v)
```

    [2 4 6] * [1 3 5] = [ 2 12 30]


#### Division
As with other arithmetic operations, division is also performed element-wise to result in a new vector of the same length.

> ** a / b = [ a1 / b1 , a2 / b2 , a3 / b3 ]**


```python
# create two arrays in numpy and divide them - print results
v1 = np.array([2,4,6])
v2 = np.array([1,3,5])
v = v1 / v2
print (v1, '/', v2, '=', v)
```

    [2 4 6] / [1 3 5] = [2.         1.33333333 1.2       ]


So we see above that `Numpy` allows on-the-fly creation of vectors and lets' us perform arithmetic operations without having to do manual element wise operations. 

### Vector Dot-Product

A vector dot-product can be thought of as a **similarity measure** between two vectors just like "Cosine similarity measure". The relation between dot product and cosine is similar to the relation between covariance and correlation - one is normalized and bounded version of another. We shall look into this with more detail later in the section. The outcome of dot-product is a single number , i.e. a scalar value that reflects how similar (or dis-similar) the input vectors are. This is called the **dot product**, named because of the dot operator used when describing the operation i.e. a**\.**b.

The dot product is calculated as follows:


> ** a . b = [ a1 \* b1 + a2 \* b2 + a3 \* b3 ]**


This is one of common operations performed on vectors(and matrices) in deep learning domain. Let's see this works in three scenarios below using numpy with `.dot()` method:


```python
# create two arrays (dis-similar) in numpy and take the dot-product - print results
v1 = np.array([2,4,6])
v2 = np.array([1,3,5])
v = v1 .dot(v2)
print (v1, '.', v2, '=', v)
```

    [2 4 6] . [1 3 5] = 44



```python
# create two arrays (somewhat-similar) in numpy and take the dot-product - print results
v1 = np.array([5,1,5])
v2 = np.array([5,5,5])
v = v1 .dot(v2)
print (v1, '.', v2, '=', v)
```

    [5 1 5] . [5 5 5] = 55



```python
# create two arrays (similar) in numpy and take the dot-product - print results
v1 = np.array([5,5,5])
v2 = np.array([5,5,5])
v = v1 .dot(v2)
print (v1, '.', v2, '=', v)
```

    [5 5 5] . [5 5 5] = 75


So we see that the output of a dot-product can be seen as a scalar value, the magnitude of  which helps us identify how similar or different the input vectors are. 

### Vector Scalar Multiplication

Vectors can be multiplied to scalar values (i.e. a single number) to get a vector in the output with same length as input vector. Each value of the vector elements is multiplied to the scalar value as shown below:

> **s \* v = [ s \* v1 , s \* v2 , s \* v3 ]**
( here s is the scalar and v is the vector )

Numpy arrays use a simple multiplication operator to achieve this as shown below


```python
# create an array  (vector)  and multiply it with a number (scalar) - print results
v1 = np.array([2,4,6])
s = 5
v = s * v1
print (s, '*', v1, '=', v)
```

    5 * [2 4 6] = [10 20 30]


Similarly, vector-scalar addition, subtraction, and division can be performed exactly in the same way.

## Summary

In this lesson we saw an introduction to vectors in python and numpy. We performed basic arithmetic operations on vectors and also some specialized operations like dot product. We shall now move to learning the same for matrices in python and numpy. 
