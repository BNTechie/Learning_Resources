# Important mathematical functions and distributions you should be aware of:

***1. Gradient descent:***
   This is crucial for optimizing machine learning models. This optimisation algorithm is used in machine learning to minimize the cost or loss function. It is essential in algorithms such as regression models and neural networks. It is a 1st order iterative algorithm for finding a local minimum of a differentiable multivariate function.
   
   The concept in brief is the following:
   
   Take repeated steps in the opposite direction of the function at the current point, because it the the direction of steepest descent. Conversely, stepping in the direction of the gradient will lead to a local maximum of that function; the procedure is then known as gradient ascent. 

   Gradient descent is not the same as the local search algorithm, even though both are iterative optimization methods.

   *****An excellent analogy from Wikipedia:*****
   
''   
The basic intuition behind gradient descent can be illustrated by a hypothetical scenario. Persons are stuck in the mountains and are trying to get down (i.e., trying to find the global minimum). There is heavy fog such that visibility is extremely low. Therefore, the path down the mountain is not visible, so they must use local information to find the minimum. They can use the method of gradient descent, which involves looking at the steepness of the hill at their current position, then proceeding in the direction with the steepest descent (i.e., downhill). If they were trying to find the top of the mountain (i.e., the maximum), then they would proceed in the direction of steepest ascent (i.e., uphill). Using this method, they would eventually find their way down the mountain or possibly get stuck in some hole (i.e., local minimum or saddle point), like a mountain lake. However, assume also that the steepness of the hill is not immediately obvious with simple observation, but rather it requires a sophisticated instrument to measure, which the persons happen to have at the moment. It takes quite some time to measure the steepness of the hill with the instrument, thus they should minimize their use of the instrument if they wanted to get down the mountain before sunset. The difficulty then is choosing the frequency at which they should measure the steepness of the hill so not to go off track.

In this analogy, the persons represent the algorithm, and the path taken down the mountain represents the sequence of parameter settings that the algorithm will explore. The steepness of the hill represents the slope of the function at that point. The instrument used to measure steepness is differentiation. The direction they choose to travel in aligns with the gradient of the function at that point. The amount of time they travel before taking another measurement is the step size.''


   
****Usage of gradient descent:****

- **Linear Regression:**
  
Minimizes the Mean Squared Error (MSE) to find the best-fit line.

- **Logistic Regression:**
  
Minimizes the log-loss to find the decision boundary between classes.

- **Neural Networks:**

Uses gradient descent to minimize the cost function, typically cross-entropy, by adjusting the weights and biases during backpropagation.

***Challenges:***

****Local vs. Global Minima:****

For non-convex functions (like those in deep learning), gradient descent may converge to a local minimum rather than a global minimum.

****Vanishing/Exploding Gradients:****

In deep networks, gradients can become very small (vanish) or very large (explode), making learning difficult. Techniques like proper initialization, batch normalization, or alternative architectures (e.g., LSTM in RNNs) can mitigate this.

------------------------------------------------------------------------------------------------------------------------------------------------------

   
   
****3. Normal Distribution****

It is one of the most important distributions in statistics because of its natural occurrence in many processes and its central role in the Central Limit Theorem. This distribution is also known as the Gaussian distribution. 

*****Features:*****

1. Belle shaped curve: The graph of this distribution is bell-shaped, with the highest point at the mean of the data, symmetric about the mean.
2. Mean, median, and mode: Mean, median, and mode of such distribution are all equal and are located at the center of the distribution.
3. Standard Deviance and Variance: The spread of the distribution is measured by the standard deviation of the curve. The larger the standard deviation, the wider the curve and vice versa.
4. Empirical Rule (68-95-99.7 Rule):

Approximately 68% of the data falls within one standard deviation of the mean.

Approximately 95% of the data falls within two standard deviations of the mean.

Approximately 99.7% of the data falls within three standard deviations of the mean.


## Mathematical Representation of the Normal Distribution

The probability density function (PDF) of a normal distribution is given by:


```math
f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}} 
```
Where:

- \( \mu \) is the mean of the distribution.
- \( \sigma \) is the standard deviation of the distribution.
- \( \sigma^2 \) is the variance of the distribution.
- \( x \) is the value for which you want to calculate the probability density.

### Standard Normal Distribution

The standard normal distribution is a special case where the mean \( \mu = 0 \) and the standard deviation \( \sigma = 1 \). Any normal distribution can be standardized using the Z-score formula:


```math
Z = \frac{X - \mu}{\sigma}

```

Where $\( Z \)$ represents the number of standard deviations a data point \( X \) is from the mean.

***Importance in statistics:***

- Central Limit Theorem(CLT):

  The CLT states that the sum or average of a large number of independent and identically distributed random variables, regardless of the original distribution, will be approximately normally distributed. This makes normal distribution invredibly important in inferential statistics.

- Hypothesis testing and confidence interval: Many statistical methods, e.g., t-tests, ANOVA, and confidence intervals, assume that the underlying data are normally distributed. 

----------------------------------------------------------------------------------------------------------------------------------------
****5. Z-score****

A Z-score or standard score is a statistical measure that describes how far a data point is from the mean of a data set in terms of standard deviations. It is a way to standardize different data points from different distributions to a common scale making it easier to compare them directly.


Uses of Z-scores:

Comparing Different Data Points:

Z-scores allow you to compare data points from different distributions, even if those distributions have different means and standard deviations.

Identifying Outliers:

Z-scores can help identify outliers in a data set. Typically, data points with Z-scores greater than +3 or less than -3 are considered outliers.

Standardizing Scores:

Z-scores standardize data, which is especially useful in statistical analysis, when combining results from different scales or distributions.

Probability Calculation:

In a standard normal distribution (a normal distribution with a mean of 0 and a standard deviation of 1), Z-scores can be used to calculate the probability of a data point occurring within a certain range. This is often done using Z-tables.


 
-----------------------------------------------------------------------------------------------------------------------------------------   
****7. Sigmoid****

The sigmoid function is a mathematical function that is commonly used in data science, particularly in machine learning and artificial intelligence. It is a type of activation function, often used in neural networks, especially in logistic regression models. The sigmoid function is valued for its ability to map any real-valued number into a range between 0 and 1, which makes it particularly useful for binary classification problems.

Mathematicall defined as,
```math
\sigma(x) = \frac{1}{1 + e^{-x}}
```

Where:

- \( x \) is the input to the function.
- \( e \) is the base of the natural logarithm, approximately equal to 2.71828.

### Key Characteristics:

1. **Output Range:** The sigmoid function outputs values in the range (0, 1). This makes it especially useful for models where outputs need to be interpreted as probabilities.

2. **S-Shaped Curve:** The function has an S-shaped curve, with values approaching 0 for large negative inputs and approaching 1 for large positive inputs.

3. **Derivative:** The derivative of the sigmoid function is:

    
```math
   \sigma'(x) = \sigma(x) \cdot (1 - \sigma(x))
   
```
   This property is useful during the backpropagation process in training neural networks.

### Example Usage in Logistic Regression:

In logistic regression, the sigmoid function is applied to the linear combination of input features to model the probability that a given input belongs to a particular class:


```math
P(Y=1|X) = \sigma(\theta^T X) = \frac{1}{1 + e^{-(\theta^T X)}}
```

Here, $ \theta^T X $ represents the weighted sum of the input features.

### Applications:

- **Binary Classification:** Converting the output of a model to a probability.
- **Neural Networks:** Introducing non-linearity into the model, allowing it to learn complex patterns.

----------------------------------------------------------------------------------------------------------------------------------------
8. Correlation
 ---------------------------------------------------------------------------------------------------------------------------------------
***10. Cosine Similarity***

Cosine similarity is a metric used in data science to measure the similarity between two non-zero vectors in an inner product space. It is widely used in various applications such as information retrieval, text mining, and machine learning, particularly in tasks like document similarity, clustering, and recommendation systems.

Cosine similarity measures the cosine of the angle between two vectors, which gives an indication of how similar the two vectors are, regardless of their magnitude. In mathematical representation, 

Cosine similarity between two vectors, 
The cosine similarity between two vectors $\mathbf{A}$ and $\mathbf{B}$ is defined as:


```math
\text{Cosine Similarity} = \cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|}

```

Where:
- $\|\mathbf{A}.\mathbf{B}\|$ is the dot product of the vectors $\mathbf{A}$ and  $\mathbf{B}$.
- $\|\mathbf{A}\|$ is the magnitude (or norm) of vector $\mathbf{A}$.
- $\|\mathbf{B}\|$ is the magnitude (or norm) of vector $\mathbf{B}$.
- $\theta$ is the angle between the two vectors.

### Interpretation

- **Cosine Similarity = 1**: The vectors are identical, meaning they point in the same direction (the angle between them is 0°).
- **Cosine Similarity = 0**: The vectors are orthogonal, meaning they have no similarity (the angle between them is 90°).
- **Cosine Similarity = -1**: The vectors are diametrically opposed, meaning they point in opposite directions (the angle between them is 180°).

***Applications:***

- Text Similarity:

In natural language processing (NLP), cosine similarity is often used to measure the similarity between two documents by representing them as vectors in a term frequency or TF-IDF (Term Frequency-Inverse Document Frequency) space.

- Recommendation Systems:

Cosine similarity is commonly used in recommendation systems to find items or users that are similar to a given item or user based on their vector representations of preferences or ratings.

- Clustering:

 In clustering algorithms like k-means, cosine similarity can be used to measure the distance between data points and centroids, especially in high-dimensional spaces.

- Image Similarity:

Cosine similarity can also be used in image processing to compare feature vectors extracted from images, helping to identify similar images.




---------------------------------------------------------------------------------------------------------------------------------------- 
12. Naive Bayes


----------------------------------------------------------------------------------------------------------------------------------------
13. MLE
------------------------------------------------------------------------------------------------------------------------------------
***15. OLS(ordinary Least Square) method***

The Ordinary Least Squares (OLS) method is implemented for estimating the parameters in a linear regression model. The objective  of OLS is to find the best-fitting line through a set of data points by minimizing the sum of the squared differences (the residuals) between the observed values and the values predicted by the linear model.

OLS is widely used in predictive modeling, economic forecasting, and inferential statistics to understand relationships between variables and make predictions based on linear trends in the data.


Assumptions:

- Linearity: The relationship between the dependent and independent variables is linear.
- Independence: Observations are independent of each other.
- Homoscedasticity: The variance of the error terms is constant across all levels of the independent variables.
- Normality: The error terms are normally distributed.


In a linear regression model, the relationship between the dependent variable \( y \) and the independent variables \( X_1, X_2, \dots, X_n \) is given by:

```math
y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_n X_n + \epsilon
```

Where:
- $ \beta_0, \beta_1, \dots, \beta_n$ are the coefficients (parameters) to be estimated.
- \( \epsilon \) is the error term or residual, representing the difference between the observed value and the predicted value.

### Objective of OLS

The goal of OLS is to find the coefficients \( \beta_0, \beta_1, \dots, \beta_n \) that minimize the sum of the squared residuals:
```math
\text{Minimize } \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
```

Where:
- $y_i$ is the observed value of the dependent variable.
- $\hat{y}_i$ is the predicted value, calculated as:
```math
\hat{y}_i = \beta_0 + \beta_1 X_{i1} + \beta_2 X_{i2} + \dots + \beta_n X_{in}
```
The coefficients estimated by OLS represent the change in the dependent variable for a one-unit change in the corresponding independent variable, holding all other variables constant.

----------------------------------------------------------------------------------------------------------------------------------------
16. F1 score


---------------------------------------------------------------------------------------------------------------------------------------
17. ***ReLU***

ReLU (Rectified Linear Unit) is one of the most widely used activation functions in deep learning and neural networks. It plays a significant role in enabling neural networks to learn complex patterns by introducing non-linearity into the model.


## ReLU (Rectified Linear Unit)

ReLU (Rectified Linear Unit) is a popular activation function used in neural networks, particularly in deep learning. It introduces non-linearity into the model, allowing neural networks to learn complex patterns.

### Mathematical Representation

The ReLU function is defined as:

```math
\text{ReLU}(x) = \max(0, x)
```

Where:
- $x$ is the input to the function.
- The output is $x$ if $x$ is greater than 0, and 0 otherwise.

### Variants of ReLU

#### 1. Leaky ReLU

Leaky ReLU is a variant of ReLU that allows a small, non-zero gradient when the input is negative:
```math
\text{Leaky ReLU}(x) = \max(\alpha x, x)
```

Where:
- $\alpha$ is a small constant (typically $\alpha = 0.01$).

#### 2. Parametric ReLU (PReLU)

Parametric ReLU is similar to Leaky ReLU, but the slope for negative inputs is learned during training:

```math
\text{PReLU}(x) = \max(\alpha x, x)
```

Where:
- $\alpha$ is a parameter that is learned during the training process.

#### 3. Exponential Linear Unit (ELU)

ELU is another variant that smooths out the output for negative inputs:

```math
\text{ELU}(x) = 
\begin{cases} 
x & \text{if } x > 0 \\
\alpha (e^x - 1) & \text{if } x \leq 0
\end{cases}
```

Where:
- $\alpha$ is a positive constant.

ReLU is the default activation function in many deep learning models, particularly in Convolutional Neural Networks (CNNs) and fully connected layers in feedforward networks, frequently used in image and text processing due to its efficiency and ability to handle large datasets, while also being integral for capturing non-linearities in tasks such as image recognition and natural language processing; however, it has limitations like the "dying ReLU" problem, where neurons can become inactive.


Limitations:

- Dying ReLU Problem:

A potential issue where neurons can become inactive and always output 0, particularly if the weights are initialized poorly or if the learning rate is too high. This can lead to parts of the network not learning at all.

- Unbounded Output:

Unlike sigmoid or tanh, ReLU does not cap its output for positive inputs, which can lead to large values and potentially unstable learning if not managed properly.



### Summary

ReLU and its variants are critical for introducing non-linearity into neural networks, enabling them to model complex patterns in data. The basic ReLU function is computationally efficient and widely used, while variants like Leaky ReLU, PReLU, and ELU offer improvements for specific scenarios, particularly in addressing issues like the "dying ReLU" problem.





-------------------------------------------------------------------------------------------------------------------------------------
***19. Softmax***

 
The softmax function is commonly used in multi-class classification tasks in machine learning. It converts a vector of raw scores (logits) into a probability distribution, where the probabilities of all classes sum to 1.

Softmax is generally used in the final layer of a neural network for multi-class classification. It transforms the logits (raw predictions) into probabilities, allowing the model to predict the probability that a given input belongs to each possible class.

### Mathematical Representation

Given a vector of scores $ z = [z_1, z_2, \dots, z_n]$, the softmax function for each element $z_i$ is defined as:


```math
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}}

```

Where:
- $z_i$ is the $i$-th element of the input vector $z$.
- $e^{z_i}$ is the exponential function applied to  $z_i$.
- The denominator $\sum_{j=1}^{n} e^{z_j}$ is the sum of the exponentials of all elements in the input vector, ensuring that the output probabilities sum to 1.

### Output

The output of the softmax function is a probability distribution, where each element represents the probability of the corresponding class. The sum of all probabilities equals 1, making it suitable for multi-class classification tasks. The class with the highest probability is typically chosen as the predicted class. The magnitude of the logits determines the "confidence" of the predictions.



-------------------------------------------------------------------------------------------------------------------------------------
20. R2 score



-----------------------------------------------------------------------------------------------------------------------------------
21. MSE


------------------------------------------------------------------------------------------------------------------------------------
22. MSE+L2 Reg


-------------------------------------------------------------------------------------------------------------------------------------
23. Eigen vectors


----------------------------------------------------------------------------------------------------------------------------------
24. Entropy


------------------------------------------------------------------------------------------------------------------------------
25. KMeans


----------------------------------------------------------------------------------------------------------------------------------
26. KL Divergence


-------------------------------------------------------------------------------------------------------------------------------------
27. Log-loss


----------------------------------------------------------------------------------------------------------------------------------------
28. SVM


------------------------------------------------------------------------------------------------------------------------------------
29. Linear Regression


-----------------------------------------------------------------------------------------------------------------------------------
***30. SVD***

Singular Value Decomposition (SVD) is a powerful linear algebra technique widely used in data science for dimensionality reduction, matrix factorization, and noise reduction, among other applications. SVD is particularly useful in areas like recommendation systems, image compression, and natural language processing.

Applications in Data Science:

- Recommendation Systems: SVD is used in collaborative filtering techniques to factorize the user-item interaction matrix, helping to identify latent features and make recommendations.
Image Compression: By applying SVD to image matrices, it's possible to store the image using fewer data points without significantly losing image quality.

- Natural Language Processing (NLP): SVD is applied in techniques like Latent Semantic Analysis (LSA) to reduce the dimensionality of term-document matrices, revealing underlying relationships between words and documents.
Advantages:

- Noise Reduction: SVD helps in eliminating noise by focusing on the most significant singular values and ignoring smaller ones, which often represent noise in the data.

- Data Compression: By reducing the dimensionality of the data, SVD enables efficient storage and faster computations.





-------------------------------------------------------------------------------------------------------------------------------------
31. Lagrange multiplier

   
