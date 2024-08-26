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
'
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

\[
Z = \frac{X - \mu}{\sigma}
\]

Where \( Z \) represents the number of standard deviations a data point \( X \) is from the mean.

'

----------------------------------------------------------------------------------------------------------------------------------------
5. Z-score

-----------------------------------------------------------------------------------------------------------------------------------------   
7. Sigmoid
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

   
