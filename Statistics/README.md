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

\[ 
f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}} 
\]

Where:

- \( \mu \) is the mean of the distribution.
- \( \sigma \) is the standard deviation of the distribution.
- \( \sigma^2 \) is the variance of the distribution.
- \( x \) is the value for which you want to calculate the probability density.

### Standard Normal Distribution

The standard normal distribution is a special case where the mean \( \mu = 0 \) and the standard deviation \( \sigma = 1 \). Any normal distribution can be standardized using the Z-score formula:

$ \[
Z = \frac{X - \mu}{\sigma}
\] $

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
\[
\sigma(x) = \frac{1}{1 + e^{-x}}
\]


```

Where:

- \( x \) is the input to the function.
- \( e \) is the base of the natural logarithm, approximately equal to 2.71828.

### Key Characteristics:

1. **Output Range:** The sigmoid function outputs values in the range (0, 1). This makes it especially useful for models where outputs need to be interpreted as probabilities.

2. **S-Shaped Curve:** The function has an S-shaped curve, with values approaching 0 for large negative inputs and approaching 1 for large positive inputs.

3. **Derivative:** The derivative of the sigmoid function is:
   \[
   \sigma'(x) = \sigma(x) \cdot (1 - \sigma(x))
   \]
   This property is useful during the backpropagation process in training neural networks.

### Example Usage in Logistic Regression:

In logistic regression, the sigmoid function is applied to the linear combination of input features to model the probability that a given input belongs to a particular class:


```math
\[
P(Y=1|X) = \sigma(\theta^T X) = \frac{1}{1 + e^{-(\theta^T X)}}
\]
```

Here, \( \theta^T X \) represents the weighted sum of the input features.

### Applications:

- **Binary Classification:** Converting the output of a model to a probability.
- **Neural Networks:** Introducing non-linearity into the model, allowing it to learn complex patterns.

--------------------------------------------------------------------------------------------------------------------------------------------
8. Correlation
9. Cosine Similarity
10. Naive Bayes
11. MLE
12. OLS
13. F1 score
14. ReLU
15. Softmax
16. R2 score
17. MSE
18. MSE+L2 Reg
19. Eigen vectors
20. Entropy
21. KMeans
22. KL Divergence
23. Log-loss
24. SVM
25. Linear Regression
26. SVD
27. Lagrange multiplier

   
