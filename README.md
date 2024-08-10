# Popular Deployment Methods for Data Scientists

Data scientists use a variety of methods to deploy machine learning models. Here are some of the most popular deployment methods, each with a brief explanation:

## 1. Flask and FastAPI

### Flask:
- Flask is a lightweight web framework for Python. It’s simple to set up and use, making it popular for deploying small to medium-sized ML models.
- **Pros**: Easy to set up, flexible.
- **Cons**: May require additional work for scaling and handling high traffic.

### FastAPI:
- FastAPI is a modern, fast (high-performance) web framework for building APIs with Python 3.6+ based on standard Python type hints.
- **Pros**: High performance, automatic generation of OpenAPI documentation, asynchronous capabilities.
- **Cons**: Slightly more complex than Flask.

## 2. Docker
- Docker is a tool designed to make it easier to create, deploy, and run applications by using containers.
- **Pros**: Consistent environments, easy to scale, portable.
- **Cons**: Requires knowledge of Docker and containerization concepts.

## 3. Kubernetes
- Kubernetes is an open-source platform designed to automate deploying, scaling, and operating application containers.
- **Pros**: Manages containers at scale, self-healing, automated rollouts and rollbacks.
- **Cons**: Complex to set up and manage.

## 4. Cloud Services

### AWS SageMaker:
- Amazon SageMaker is a fully managed service that provides every developer and data scientist with the ability to build, train, and deploy machine learning models quickly.
- **Pros**: Fully managed, integrates well with other AWS services.
- **Cons**: Can be expensive, has a learning curve.

### Google AI Platform:
- Google AI Platform is a managed service that enables you to easily build, deploy, and scale machine learning models.
- **Pros**: Fully managed, integrates well with Google Cloud services.
- **Cons**: Can be expensive, has a learning curve.

### Azure Machine Learning:
- Azure Machine Learning is a cloud-based environment that you can use to train, deploy, automate, manage, and track ML models.
- **Pros**: Fully managed, integrates well with Azure services.
- **Cons**: Can be expensive, has a learning curve.

## 5. Serverless Architectures

### AWS Lambda:
- AWS Lambda lets you run code without provisioning or managing servers. You pay only for the compute time you consume.
- **Pros**: No server management, automatic scaling.
- **Cons**: Limited execution time (15 minutes max), stateless.

### Google Cloud Functions:
- Google Cloud Functions is a serverless execution environment for building and connecting cloud services.
- **Pros**: No server management, automatic scaling.
- **Cons**: Limited execution time, stateless.

## 6. Model-as-a-Service Platforms
- Platforms like Algorithmia, MLflow, and TensorFlow Serving provide infrastructure and tools specifically for deploying and managing machine learning models.
- **Pros**: Designed for ML models, provide model management tools.
- **Cons**: Can be expensive, may require integration with other tools.
# Machine Learning Algorithms:

_***XGBoost**

XGBoost (eXtreme Gradient Boosting) is a powerful and scalable machine learning algorithm that is particularly well-suited for structured/tabular data. It is an implementation of gradient-boosted decision trees designed for speed and performance.

Here are some key points about XGBoost:

Boosting Algorithm: XGBoost is a boosting algorithm, which means it builds an ensemble of trees sequentially. Each tree attempts to correct the errors of the previous trees, improving the overall performance of the model.

Gradient Boosting: It uses gradient boosting, which involves optimizing a loss function (such as mean squared error for regression or log loss for classification) by adding new trees that predict the residuals (errors) of the previous trees.

Efficiency and Speed: XGBoost is known for its efficiency. It includes several optimizations such as:

Parallel processing
Tree pruning to avoid overfitting
Efficient handling of sparse data
Cache awareness and out-of-core computation for large datasets
Regularization: XGBoost includes regularization terms to control the complexity of the model, which helps in preventing overfitting. The parameters alpha and lambda (L1 and L2 regularization) help in achieving this.

Flexibility: It supports various objective functions, including regression, classification, and ranking. It also allows custom objective functions and evaluation metrics.

Cross-Validation: XGBoost provides built-in cross-validation capabilities, making it easy to tune hyperparameters and evaluate model performance.

Feature Importance: It can calculate feature importance scores, which helps in understanding the impact of each feature on the model's predictions.

Wide Adoption: XGBoost has been widely adopted in machine learning competitions and industry applications due to its robustness and superior performance. It has been the algorithm of choice in many winning solutions of data science competitions like Kaggle.

### Youtube Video: https://www.youtube.com/watch?v=OtD8wVaFm6E

# What is Quasi-Newton method ?
Quasi-Newton methods are optimization algorithms used to find the minimum (or maximum) of a function. These methods belong to the broader class of quasi-Newton optimization algorithms, which are designed for unconstrained optimization problems. The primary goal is to iteratively find the minimum of a scalar function without requiring the calculation of derivatives at each step.

Quasi-Newton methods are iterative and build an approximation to the inverse Hessian matrix of the objective function. The Hessian matrix represents the second-order partial derivatives of the objective function with respect to its variables and plays a crucial role in determining the curvature of the function.

The Newton-Raphson method is a classical optimization algorithm that uses the exact Hessian matrix at each iteration. However, computing the exact Hessian can be computationally expensive and, in some cases, impractical. Quasi-Newton methods offer a more computationally efficient alternative by approximating the Hessian matrix.

The Broyden–Fletcher–Goldfarb–Shanno (BFGS) algorithm and the Davidon–Fletcher–Powell (DFP) algorithm are two well-known examples of quasi-Newton methods. These methods iteratively update their estimates of the inverse Hessian based on the changes in the gradient of the objective function.

Here's a brief overview of the BFGS algorithm, one of the most widely used quasi-Newton methods:

Initialization:

Start with an initial guess for the minimizer and an initial approximation to the inverse Hessian matrix.
Iteration:

At each iteration, compute the gradient of the objective function and update the approximation to the inverse Hessian.
Search Direction:

Determine the search direction by multiplying the negative inverse Hessian approximation by the gradient.
Line Search:

Perform a line search along the search direction to find an optimal step size.
Update:

Update the current estimate of the minimizer using the step size.
Convergence:

Check for convergence criteria, such as a small change in the objective function or gradient.
Repeat:

If convergence criteria are not met, repeat the process.
Quasi-Newton methods are particularly useful when the exact Hessian is challenging to compute, and they have been successfully applied to various optimization problems in fields such as machine learning, numerical optimization, and mathematical modeling. The efficiency of quasi-Newton methods lies in their ability to provide a good approximation of the Hessian without the computational cost of its exact calculation.







https://www.youtube.com/watch?v=UvGQRAA8Yms


# Biological concepts: 

## What is 'housekeeping gene'?

Housekeeping genes are genes that are essential for the maintenance of basic cellular functions and are typically expressed in all cells of an organism under normal and healthy conditions. They perform fundamental roles in the upkeep of cellular physiology and survival. Here are some key points about housekeeping genes:

### Key Characteristics of Housekeeping Genes:

1. **Essential Functions**: Housekeeping genes are involved in crucial cellular processes such as energy production, metabolism, cell structure maintenance, DNA repair, and protein synthesis.

2. **Constitutive Expression**: These genes are usually expressed at relatively constant levels across different cell types and conditions because their protein products are required continuously for the cell to function properly.

3. **Universal Presence**: Housekeeping genes are present in all cells of an organism, irrespective of the tissue type or developmental stage. They are fundamental to the basic operations of every cell.

4. **Stable Expression Levels**: The expression levels of housekeeping genes are relatively stable, making them reliable internal controls in various experimental settings, such as quantitative PCR and gene expression studies.

### Examples of Housekeeping Genes:

- **GAPDH (Glyceraldehyde-3-phosphate dehydrogenase)**: Involved in glycolysis, the process of breaking down glucose to produce energy.
- **ACTB (Beta-actin)**: Part of the cytoskeleton, playing a critical role in cell structure and integrity.
- **RPLP0 (Ribosomal protein, large, P0)**: A component of the ribosome, essential for protein synthesis.
- **HPRT1 (Hypoxanthine-guanine phosphoribosyltransferase)**: Involved in nucleotide synthesis and metabolism.
- **B2M (Beta-2-microglobulin)**: Part of the major histocompatibility complex (MHC) class I molecule, important for immune response.

### Importance in Research:

- **Normalization Controls**: Due to their stable expression, housekeeping genes are often used as reference genes to normalize data in gene expression studies, ensuring that variations in experimental conditions do not affect the results.
- **Cellular Health Indicators**: Consistent expression of housekeeping genes is an indicator of normal cellular function and health, whereas deviations can signal cellular stress or pathology.

In summary, housekeeping genes are essential for the fundamental operations of cells, consistently expressed to support critical cellular functions, and serve as vital tools in molecular biology research for ensuring accurate and reliable experimental results.



## Algorithm and Datasctructure

https://www.youtube.com/watch?v=8hly31xKli0

# Website for specific tools:

Plotting Upsetplot: https://jokergoo.github.io/ComplexHeatmap-reference/book/upset-plot.html


# Machine_Learning_Resource

I list down some of the resource links I often come across and take help from for my  research.
1. https://github.com/louisfb01/Best_AI_paper_2020
2. Machine Learning Mastery by Jason - https://machinelearningmastery.com/ (This one is my favourite.)
3. Probability concepts explained: Maximum likelihood estimation (https://towardsdatascience.com/probability-concepts-explained-maximum-likelihood-estimation-c7b4342fdbb1)
4. Some intuitive questions on Data Science: https://career-accelerator.corsairs.network/99-questions-every-entry-level-analyst-should-be-able-to-answer-68cb45f9c91a

5.https://whats-ai.medium.com/top-10-computer-vision-papers-2020-aa606985f688

6.https://github.com/louisfb01/Top-10-Computer-Vision-Papers-2020

7. https://medium.com/towards-artificial-intelligence/start-machine-learning-in-2020-become-an-expert-from-nothing-for-free-f31587630cf7 by  Louis (What’s AI) Bouchard


## Sequence alignment with BWA : https://www.youtube.com/watch?v=1wcFavYt6uU

Deep learning
=============
- Tensorflow: https://github.com/tensorflow/tensorflow
- TFLearn: https://github.com/tflearn/tflearn
- PyTorch: https://github.com/pytorch/pytorch
- Apache MXNET: https://github.com/apache/incubator-mxnet
- Theano: https://github.com/Theano/Theano
- Caffe: https://github.com/BVLC/caffe and
- Caffe2: https://caffe2.ai/docs/tutorials.html
- Fast AI: https://github.com/fastai/fastai
- CNTK: https://github.com/Microsoft/CNTK
- Lasagne: https://github.com/Lasagne/Lasagne
- Chainer: https://github.com/chainer/chainer
- Nolearn: https://github.com/dnouri/nolearn
- Elephas - https://github.com/maxpumperla/elephas
- Spark deep learning: https://github.com/databricks/spark-deep-learning
- Keras: https://github.com/keras-team/keras


Bayesian inference/Advanced Statistics/Probabilistic models
===========================================================
All MCMC/SMC pacakges: https://gabriel-p.github.io/pythonMCMC/
Bayesian deep learning: https://zhusuan.readthedocs.io/en/latest/
Duke: https://people.duke.edu/~ccc14/sta-663/MCMC.html
BayesPy: http://bayespy.org/examples/examples.html
Statsmodels: https://github.com/statsmodels/statsmodels
XGBoost: https://github.com/dmlc/xgboost
LightGBM: https://github.com/Microsoft/LightGBM
Catboost: https://github.com/catboost/catboost
PyBrain: https://github.com/pybrain/pybrain
Eli5: https://github.com/TeamHG-Memex/eli5


Deep Learning- Variational autoencoders
========================================
1. https://zhusuan.readthedocs.io/en/latest/tutorials/vae.html Variational autoencoders
2. https://github.com/kvfrans/variational-autoencoder.
3. https://wiseodd.github.io/techblog/2016/12/10/variational-autoencoder/


Python big data
================
Parallel computing: https://wiki.python.org/moin/ParallelProcessing
GPU Compatbilit6y: https://docs.anaconda.com/anaconda/user-guide/tasks/gpu-packages/
PyCUDA: https://documen.tician.de/pycuda/tutorial.html
PyGPU: http://fileadmin.cs.lth.se/cs/Personal/calle_lejdfors/pygpu/
AWS: https://aws.amazon.com/developer/language/python/ (see sample code and 10 mins tutorial)
Apache Spark: https://spark.apache.org/docs/0.9.1/python-programming-guide.html
PySpark: https://spark.apache.org/docs/0.9.1/python-programming-guide.html
Apache Hadoop: https://hadoop.apache.org/


Reinforcment Learning
======================
https://github.com/keras-rl/keras-rl
OpenAI: https://github.com/openai/gym


Optimization
============
Convex: https://cvxopt.org/
Platypus: https://platypus.readthedocs.io/en/latest/
PyGMO: http://esa.github.io/pygmo/
DEAP: https://deap.readthedocs.io/en/master/examples/index.html
GAFT: https://github.com/pytlab/gaft



## Some short reads:

5 Beginner-Friendly Steps to Learn Machine Learning and Data Science with Python — Daniel Bourke
What is Machine Learning? — Roberto Iriondo

Machine Learning for Beginners: An Introduction to Neural Networks — Victor Zhou

A Beginners Guide to Neural Networks — Thomas Davis

Understanding Neural Networks — Prince Canuma

Reading lists for new MILA students — Anonymous

The 80/20 AI Reading List — Vishal Maini



##Some useful youtube link for simple demonstartion of ML topics:

1.https://www.youtube.com/watch?v=8HyCNIVRbSU---LSTM 


#Interview Questions:

1.https://www.analyticsvidhya.com/blog/2020/04/comprehensive-popular-deep-learning-interview-questions-answers/


2.https://intellipaat.com/blog/interview-question/deep-learning-interview-questions/ 


3. https://www.edureka.co/blog/interview-questions/machine-learning-interview-questions/. from Edureka

4. https://medium.com/modern-nlp/nlp-interview-questions-f062040f32f7 -- medium by 

Pratik Bhavsar

5. A simplictic way of understanding transformer model in NLP :https://www.analyticsvidhya.com/blog/2019/06/understanding-transformers-nlp-state-of-the-art-models/

6. A brief discussion on BERT: https://towardsdatascience.com/understanding-bert-bidirectional-encoder-representations-from-transformers-45ee6cd51eef

AI conference paper and presentation list: https://crossminds.ai/explore/

7. An excellent resource for text similarities in NLP

8. What is CNN? Convolutional Neural Networks: The Biologically-Inspired Model

https://www.codementor.io/@james_aka_yale/convolutional-neural-networks-the-biologically-inspired-model-iq6s48zms

9 AI in drug discivery

https://practicalcheminformatics.blogspot.com/2021/01/ai-in-drug-discovery-2020-highly.html

10. An interesting github page on Data Science and Machine Learning: https://github.com/achuthasubhash/Complete-Life-Cycle-of-a-Data-Science-Project



## Bayesian Neural Network

1. https://www.google.com/url?q=https://analyticsindiamag.com/hands-on-guide-to-bayesian-neural-network-in-classification/&sa=D&source=hangouts&ust=1620269400965000&usg=AFQjCNEteFofBga-tgHNRzxZraQwPYYWEA

2. https://keras.io/examples/keras_recipes/bayesian_neural_networks/



### How Can You Distinguish Yourself from Hundreds of Other Data Science Candidates?




https://towardsdatascience.com/how-to-distinguish-yourself-from-hundreds-of-data-science-candidates-62457dd8f385


## Good blog post on NLP problem solving




# Machine Learning in Bioinformatics

1.https://www.kdnuggets.com/2019/09/explore-world-bioinformatics-machine-learning.html

2.https://medium.com/@alenaharley/tumor-classification-using-gene-expression-data-poking-at-a-problem-using-fast-ai-again-8633c2256c85


# Pathway analysis

## Pathway enrichment analysis of metabolites

1. Lilikoi: an R package for personalized pathway-based classification modeling using metabolomics data 
   (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6290884/)
   
   
# Feataure importance :

https://machinelearningmastery.com/calculate-feature-importance-with-python/

# Phylogenetic association analysis related resources:

1.https://dendropy.org/programs/sumtrees.html


# AutoML Resources:

1. H2O AutoML - https://lnkd.in/gcxQSEW2
2. AutoGluon - https://lnkd.in/gXcrqnU9
3. AutoKeras - https://lnkd.in/ghsjphDt
4. Auto-PyTorch - https://lnkd.in/gbbNQy5R
5. Auto-sklearn - https://lnkd.in/g4MxeeVT
6. EvalML - https://lnkd.in/gDjQX3At
7. FLAML - https://lnkd.in/gUkiwqyb
8. LightAutoML - https://lnkd.in/gU2-jccZ
9. MLJAR - https://mljar.com/
10. PyCaret AutoML - https://lnkd.in/gvw8DNv8
11. TPOT - https://lnkd.in/g3z9YtuU
12. GradsFlow - https://docs.gradsflow.com/en/latest/

A notebook by Rohan Rao with examples on  the above mentioned tools/libraries.

https://www.kaggle.com/rohanrao/automl-tutorial-tps-september-2021


## Some online resources for motivation

1. Kobe Bryant: https://www.youtube.com/watch?v=VSceuiPBpxY
2. Bollywood Actor Anupam Kher with Gaur Gopal Das Best Indian Motivational Speaker
3. https://www.youtube.com/watch?v=DGIjuVbGP_A

## Videos on ML and healthcare

1.https://www.youtube.com/watch?v=oyVnONlEZoA
2. 

## Some free youtuber sharing how to learn DataScience:
1. Ken Jee: https://www.youtube.com/c/KenJee1
2. Dhaval Patel: https://www.youtube.com/playlist?list=PLeo1K3hjS3us_ELKYSj_Fth2tIEkdKXvV
3. Tina Huang: https://www.youtube.com/channel/UC2UXDak6o7rBm23k3Vv5dww
4. Andrew Mo: https://www.youtube.com/channel/UC23emuGbNM7twofQIrEgPBQ
5. 

## Some blogs on Statistics:
- https://learningstatisticswithr.com/
- https://advstats.psychstat.org/book/power/index.php (Online book for Free with example codes in R, I found it handy.)
- https://worthylab.org/statistics/
- https://r4ds.had.co.nz/
- https://xcelab.net/rm/statistical-rethinking/

## Multivariate Regression : https://library.virginia.edu/data/articles/getting-started-with-multivariate-multiple-regression
## What is odd ratio in exact test? 
In statistics, especially in the context of hypothesis testing, the odds ratio (OR) is a measure of association between an exposure and an outcome. It quantifies the strength and direction of the relationship between two variables. The odds ratio is often used in logistic regression analysis and in studies where the outcome of interest is binary (e.g., success or failure, presence or absence).

In the context of an exact test, such as Fisher's exact test, the odds ratio is used to compare the odds of an event (e.g., having a certain characteristic or outcome) between two groups. Fisher's exact test is used to determine if there is a significant association between two categorical variables by examining the relationship between their frequencies.

Here's how the odds ratio is typically calculated in the context of Fisher's exact test:

- **For a 2x2 contingency table**: If you have a 2x2 table where rows represent two groups (e.g., treatment and control) and columns represent the presence or absence of an outcome (e.g., success or failure), the odds ratio is calculated as the ratio of the odds of success in one group to the odds of success in the other group.

- **Formula**: Let's say the 2x2 table looks like this:

  ```
          Outcome Present   Outcome Absent
  Group A      a                 b
  Group B      c                 d
  ```

  Then the odds ratio (OR) is given by:
  
  \[ \text{OR} = \frac{ad}{bc} \]

- **Interpretation**: An odds ratio greater than 1 indicates that the event (e.g., success) is more likely to occur in the first group compared to the second group. An odds ratio less than 1 indicates the opposite. An odds ratio of 1 suggests that there is no association between the exposure and the outcome.

In Fisher's exact test, the p-value associated with the odds ratio is used to determine if the observed association between the two variables is statistically significant. If the p-value is below a predetermined significance level (often 0.05), it indicates that the observed association is unlikely to have occurred by chance alone, and there is evidence of a significant relationship between the variables.


# Some-error-in-R-packages-and-how-to-resolve
This repository is a collection of errors we very often encounter while running R packages. Sometimes there are a few points to be taken into account to get rid of the errors. I have noted down problems I have encountered over the years.

## boundary (singular) fit: see help('isSingular') while running glmer function.

When you receive the warning message "boundary (singular) fit" in the context of a glmer (generalized linear mixed-effects model) in R, it indicates that there may be a singularity issue in the model. Singular fits can arise from various issues, such as overparameterization, multicollinearity, or an insufficient number of observations relative to the complexity of the model.

Here are some considerations when encountering this warning:

### Understand the Warning:

The warning suggests that there might be problems with the estimation of the model parameters due to singularity. It's a signal that the model might be too complex for the available data.
### Evaluate Model Fit:

Assess the overall fit of your model. You can use diagnostic tools, such as residual plots, to check how well the model captures the patterns in the data.
### Check Model Stability:

If your model is unstable due to singularity, the estimated coefficients may be unreliable. Unstable models can lead to wide confidence intervals and difficulties in making valid inferences.
Examine Variable Importance:

Consider the importance of the variables included in your model. If the singularity is related to a specific variable or set of variables, evaluate whether they are crucial for your research question.
Simplify the Model:

Review the model structure and simplify it if needed. Remove non-essential fixed or random effects, or reconsider the inclusion of certain variables.
### Investigate Collinearity:

Check for collinearity among predictors. High collinearity can lead to singular fits. Examine the correlation matrix or calculate variance inflation factors (VIFs).
Review Grouping Structure:

If you have random effects, assess the grouping structure. Too many random effects for a small number of groups can lead to singularity issues.
Consult Statistical Literature or Experts:

If the warning persists and you're uncertain about the appropriate steps to take, consult statistical literature or seek advice from experts in the field.
Check for Convergence:

Verify if the model has converged. Sometimes, a lack of convergence can contribute to singular fits. You can check convergence using the convergence information in the model summary.

# GitHub profile: 


[https://github.com/BNTechie/GWAStutorial](https://github.com/monogenea/GWAStutorial)

