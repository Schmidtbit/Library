
# The Elements of Data Science
This course is available on the [AWS Machine Learning Certification Study Path](https://aws.amazon.com/training/learning-paths/machine-learning/exam-preparation/)

## What is Data Science?
 Processes and systems to extract knowledge or insights from data, either structured or unstructured. The Data Scientist manages, analyzes, and visualizes data to support the Machine Learning workflow.

#### What is Machine Learning?
Set of algorithms used to learning from large amounts of input data and produce predictions, prescriptions, and inferences.

* Training Data: a large set of high-quality labeled examples
* Goal: Given a training set, find a function that best describes the label.  

#### Types of Machine Learning:
* Supervised Learning (data has true labels) - regression & classification
* Semi-supervised learning (data has _some_ true labels)
* Unsupervised Learning (data is grouped) - clustering algorithms
* Reinforcement learning (reward and penalty based on desired outcome)

#### ML Workflow
1. __Problem Formulation:__ What business problem do you want to solve? Wha data is available? The Data Science and ML workflow always starts with data collection.
2. __Data Preparation & Pre-processing:__
  * removing outliers
  * handling missing values
  * cleaning and feature engineering
3. __Data Modeling:__
  * Model Training & evaluation
  * Hyper-parameter Tuning and debugging
4. __Prepare Model for Production__

#### Data Quality
* Consistency of the Data
* Accuracy of the Data
* Noisy Data
* Missing Data
* Outliers
* Bias
* Variance

#### Overfitting = High Variance = failure to generalize
Model performs well on training but poorly on test Set. Indicates model may be capturing too much noise in the data. Maybe there are too many features.

#### Underfitting = High Bias = failure to identify important patterns
Model is too simple or there are too few explanatory variables.

#### Linear Regression
* Learn weights by applying stochastic gradient descent to minimize loss function
* Good starting point due to its simplicity
* univarate = 1 variable
* multivariate = multiple variables
* Error = distance between the data point and predicted values
* The goal is to minimizes RMSE (root mean square error)
* The regressors should have a gaussian distribution with mean = 0 and fixed Variance
* multi-colinearity occurrs when multiple variables are correlated, which can lead to high variance and overfitting.

#### Logistic Regression
* Outcomes are binary (1 or 0)
* Predicts the probability of an observation belonging to a binary class.
* Sigmoid curve (probability function in logistic regression): sigma(x) = 1 / (1 + e^-x)
* Logit Function: logit(p) = log(p / (1-p))
* Prone to Outliers
* Is the data linearly separable? If yes, then you can use logistic regression. If no, you still may be able to use logistic regression depending on the shape of the separation.

#### Problem Formulation & EDA
* What is the problem you are trying to solve?
* What is the business metric? (measuring the impact of the problem)
* Is ML the appropriate approach? (maybe simple coding or RPA is appropriate)
* What data is available?
* What type of ML problem is it? (Supervised, unsupervised, reinforcement)
* What are your goals? (define the criteria for a successful outcome)

#### Sampling data
* Sample needs to be representative of the expected population (_unbiased_)
* Random Sampling (issue: rare populations can be under-represented)
* Stratified Sampling = taking a random sample from each sub-population separately
* Seasonality: when the timing of sampling matters. Use stratified sampling to counteract this.
* Trends: data shifts over time
* Data Leakage:
   - train/test bleed = when training data accidentally gets sampled into the validation dataset.
   - Leakage = when information not available in production is used to train your algorithm.

Is your model a causal model or a correlation model? Random sampling will produce a causal model. Non-random sampling will produce a correlation model. Asking if a model can be generalized is asking if the model is sensitive to the entire population of inputs.

## EDA

__Domaine Knowledge__
* AWS ML specialist
* AWS Professional Services
* AWS ML Solutions lab: Pairs your team with AWS experts to brainstorm, prepare data, build models, and train
* AWS Partner Network


__Data Sources__
* CSV file
* Traditional Database
* Amazon Glacier (object based)
* Amazon RDS (structured)
* Amazon DynamoDB (unstructured)
* Amazon Redshift (managed service)

__Data Statistics__
* dimensions (rows and columns)
* Attribute Stastics:
  - numberic data: `df.describe()` in pandas
  - categorical data: histograms, mode, most/least frequent values, percentage, number of unique values (`df["attribute_name"].value_counts()` in pandas or `distlot()` in seaborn)
  - Target data: `df["attribute_name"].value_counts()` in pandas, or `np.bincount()` in numpy
* Multivariate Statistics: Correlation, contingency tables
* Plots:
   - Density plots
   - histograms
   - boxplots
   - scatterplots (`df.plot.scatter(x=,y=)` and `pd.scatter_matrix(df)` in pandas)

__Correlated Data__
*  How to calculate correlation: use a correlation matrix where 1 is 100% correlated and 0 is no correlation whatsoever. Correlations of -1 are negatively correlated.
* Correlation Matrix Heat map: plot the `heatmap` in pandas or python where `heatmap = np.corrcoef(df.values.T)`

__Data Issues__
* missing data
* noisy data
* data on different scales
* mixed type data
* imbalanced Data
* sample Bias
* outliers

#### Feature Engineering

__Encoding Categorical Variables__
* Categorical or "discrete" variables (i.e. red, green, blue). Pandas has a data dype `dtype="category"``
 - Ordinal Categorical Variables = categories that are ordered (i.e. Small, Medium, Large) --> __To Encode:__ use the `map` function in pandas: `mapping = dict({}), df[col] = df[col].map(mapping)`
 - Nominal Categorical Variables = categories without order (i.e. red, green, blue) --> __To Encode:__ use one-hot encoding with `sklearn.preprocesing.OneHotEncoder` or `pandas.get_dummies()` in pandas.
* To encode binary ___labels___ use `sklearn.preprocessing.LebalEncoder`. If label has more than 2 classes, do not use this method.


__Handling Missing Values__

Sources of Missing Values: Undefined Values, data collection errors, left joins, etc....

Many ML Algorithms can't handle missing Values

Check Missing Values with pandas:
`df.isnull().sum()` for each column and `df.isnull().sum(axis=1)` for each row

Drop Missing Values:
`df.dropna()` to remove all rows with missing values or `df.dropna(axis=1)` to drop columns with missing values

___WARNING:___ When dropping null values your dataset may lose too much data or introduce bias into the data. Before dorpping missing values, ask:
1. What are the mechanisms that caused the missing values?
2. Are these missing values missing at random?
3. Are there rows or columns missing that you are not aware of?

Impute Missing Values:
Technique that replaces missing values with an estimated value (i.e. mean, median, most frequent)

Advanced Methods of Imputing:
* MICE (Multiple Imputation by Chained Equations) - `sklearn.impute.MICEImputer`
* Python Package `fancyimpute`
  - KNN Impute
  - Softimpute
  - MICE
  - etc...

__Feature Engineering__

* Sklearn: `sklearn.feature_extraction`
* Rules of thumb:
  - use intuition
  - try and generate many features first ___then___ apply dimensionality Reduction if needed
  - Consider squaring `X^2`, multiplication `x * y`
  - Don't overthink or include too much manual logical


__Filtering & Scaling__
* examples: remove color channels from an image or remove frequencies from a sound sample.
* Some Algorithms are sensitive to features being on different scales (i.e. gradient descent and kNN). Algorithms ___NOT___ sensitive to scale are decision trees.
  - example of a scale problem: the number of bedrooms in a house ranges from 1 to 3 but the proce ranges from $325k to $1.9M. to fix this we convert all variables to the same scale.

Common scaling transformations available in `sklearn` (performed on columns):
    * Mean/Variance standardization: `sklearn.preprocessing.StandardScaler` centers data around the mean at mean = 0 with standard deviation = 1. This method dampens the effect of outliers.
    * MinMax Scaling: `sklearn.preprocessing.MinMaxScaler` which scales values such that the minimum = 0 and maximum = 1. Advantage is that it is robust when the standard deviation is small for a data column.
    * Maxabs scaling: This method divides each value in the column by the maximum absolute value for that feature. Available in Sklearn with `sklearn.preprocessing.MaxAbsScaler`
    * Robust scaling: Find the median, 75th quartile, and the 25th quartile then calculate the new scaled value of the feature by subtracting the median and then dividing by (75th quantile - 25th quantile). This is available with `sklearn.preprocessing.RobustScaler`

The _normalization procedure_ is done for each row independently and resales to unit norm based on L1, L2, and Max norm. Widely used in text analysis and available with `sklearn.preprocesing.Normalizer`

__Polynomial Transformations & Radial Bias Function__

When you have multiple numerical values in your features you can do polynomial transformations for each of them. You can use `sklearn.preprocessing.PolynomialFeatures`

___Warning:___ beware of overfitting if the degree of polynomial transformations is too high. Consider non-polynomial transformations as well (i.e. log transformations and Sigmoid transformations) There is also a risk of extrapolation beyond the range of the data when using polynomial transformations.

The radial basis function is transformed through a center `c` and are widely used in SVM algorithms and in Radial Basis Neural Networks (RBNNs). Gaussian RBF is the most common RBF used.

__Text-Based Features__
Clean text and convert to numerical Values

1. _Bag-of-words model:_ creates a vector of numbers, one for each word (tokenize, count, and normalize). Sparse Matrix implementation is typically used and ignores relative position of words.Can be extended to bag of n-grams of words or of characters
2. _Count Vectorizer:_ Per-word value is "count" (also called term frequency). This includes lowercasing and tokenization on white space and punctuation. `sklearn.feature_extraction.text.CountVectorizer`
3. _TfidfVectorizer:_ term frequency multiplied by the inverse document frequency resulting in a per-word value downweighted for terms common across all documents (i.e. "the") `sklearn.feature_extraction.text.TfidfVectorizer`

## Model Training, Tuning, and Debugging

### Supervised Learning: Neural Networks

What is a Neural Net?
A Neural Net consists of ___perceptron nodes___ configured into _layers_. Each node is a multivariate linear function with a univariate nonlinear transformation function. Neural Nets are trained via stochastic gradient descent.

Properties:
* difficult to interpret
* expensive to train
* fast to predict   

__Perceptron__ = Most simple NN with only one layer. Input vectors and a bias vector (a bias vector is similar to the intercepts in linear regression) are linearly combined into a equation equation, which is then applied to an activation function (such as Sigmoid(z) = 1 / (1 + e^-z)).

In Scikit Learn: `sklearn.neural_network.MLPClassifier`

Deep learning Frameworks:
* MXNet
* TensorFlow
* Caffe
* PyTorch

__Convolutional Neural Network__ = Good for image classification. This is a feed-forward NN and cannot handle sequential data. Input data is assumed to be independent from one another.

input >> Convolution Layer 1 >> Pooling Layer 1 >> Convolution Layer 2 >> Pooing Layer 2 >>> ... >> Fully Connected Layer >> Output

Pooling = dimentionality reduction

__Recurrent Neural Network__ = Good for NLP and Time Series where the sequence of inputs has meaning. Input layer and output layer are connected.

### Supervised Learning: kNN
Figure out the respone of a new observation based on how similar it is to data  in the training set.
1. Define a distance metric (i.e manhattan distance)
2. Define k (the distance value) - commonly used k = sqrt(N)/2 where N is the number of sample observations.
2. Count the number of observations from each class in the training data that is within the specified distance k. The class with the highest number of observations is then assigned to the new observation.

Properties:
* KNN is non-parametric (not defined by a fixed set of parameters)
* instance based or "lazy" learning
* requires keeping original data Set
* space complexity and prediction time complexity grow with size of training data.
* consumes a lot of memory
* suffers from the __curse of dimensionality__ where points become increasingly isolated with more dimensions of a fixed size training dataset

Scikit Learn: `sklearn.neighbors.KNeighborsClassifier`

### Supervised Learning: Linear & Nonlinear Support Vector Machines (SVMs)

Linear SVMs: Data is linearly separable with a hyperplane that is defined by maximizing distance (margin) between the hyperplane and the observations.
* popular approach in research
* maximum margin separation is not possible is the data is non-seperable.

Nonlinear SVMs: uses a "kernelized" approach.
1. Choose a distance function called a "kernel"
2. map the learning task to a high dimension space
3. apply a linear SVM classifier in the new space

* not memory efficient because it stores the support vectors, which grow with the size of the training data
* computation is expensive

scikit Learn: `sklearn.svm.SVC`

### Supervised Learning: Decision Trees and Random Forests
Decision trees _split_ on feature nodes based on ___Entropy___ or the amount of "disorder in the data". Nodes are chosen based on the Information Gain (IG) associated with that feature. IG is measured by calculating entropy before and after splitting.
* entropy = 0: all samples belong to the same class
* entropy = 1: sample contains all classes in equal proportion (chaos)
The splitting procedure iterates until all "child" or "leaf" nodes are pure or entropy = 0.

Features & Properties:
* You train a tree by maximizing IG to choose splits such that the purity of the splits sets are lower
* easy to interpret
* expressive = flexible
* less need for feature transformations and can handle missing data
* susceptible to overfitting
* "Pruning" the tree is necessary in order to reduce potential for overfitting

Scikit Learn: `sklearn.tree.DecisionTreeClassifier`

__Ensenble Method: "Random Forest"__
Learn multiple models and combine their results via majority vote or averaging

* set of decision trees learned from a different randomly sampled subset with replacement
* features to split on for each tree are randomly selected subset from original features
* the prediction is the average output from all the trees
* This method increases diversity through random selection of training data and random selection of features
* reduces Variance
* each tree does not require pruning
* expensive to train & run

Scikit Learn: `sklearn.ensemble.RandomForestClassifier`

### Unsupervised Learning with k-means and Hierarchical clustering
 There is no one target variable. Our aim is only to explore the features and create our own grouping.

 __K-Means Clustering:__
1. Define the number of clusters and randomly assign the position of the clusters
2. Calculate the distance between each data point and the cluster center.
3. Assign each data point to the nearest cluster.
4. Then re-calculate the cluster center such that the center has the optimal minimum distance between all data points assigned to the cluster.
5. Repeat this process until there is no more change in ownership between clusters

Features & Properties:
* guaranteed to converge to local optimum
* suffers from ___curse of dimensionality___
* user must determine or provide the number of clusters

Scikit Learn: `sklearn.cluster.kmeans`

How to determine the optimal number of clusters (if the business use-case does not already define it)? Use the __Elbow Method__ and plot the SSE against the number of clusters. The "elbow" point on the line will identify the optimal number of clusters to use

Determine the Error of the CLustering Algorithm:
The Error is defined as SSE (Sum of Squared Error) where SSE is the squared sum of the differences between each observation belonging to the cluster and the cluster center (summed across all clusters)

__Hierarchical Clustering__
Two Types:
1. Agglomorative = bottom up approach that starts with each point in its own cluster
2. Divisive = Top down approach that starts with each point belonging to a single cluster.

Features & Properties:
* produces a dendrogram plot
* User does not need to provide the number of clusters, but does have to determine where to "cut" the dendrogram

### Model Tuning & Validation
1. Split the data into Train & Test datasets
2. Train the model with the Train Data and validate with the Test Data.
3. Tune the hyperparameters and/or try alternative algorithms and re-tain 7 re-test.

If you train and test too many times you can overfit the model. The solution is the split the Train Data again, into Train and Validation Data. The Validation Data is used during debugging and tuning while the Test Data is reserved to evaluate and measure the generalization of the final model.

Problems with Train/Test Splits:
* splitting my make the training data too small
* Solution: use a holdout method to get the test set, then use k-fold cross validation on the training set for debugging and tuning.


### Bias - Variance Tradeoff
In ML, we aim to develop models that are low bias and low variance.

__Bias__: Errors from flawed assumptions in the model. (__underfitting__)
__Variance__: Errors from over sensitivity to noise in the data causing the algorithm to model unimportant features. (__overfitting__)

--> High Bias is when a large change in input does not result in an appropriate change in output. (Not flexible enough or too simple)
--> High Variance is when a small change in input results in a large change in output. (Too flexible or too complex)

Using the ___Learning Curve___ to Determine Validation Accuracy:
Detect if the model is underfiting or overfiting and the impact of the size of the training dataset on the error.
--> Plot the training dataset and validation dataset error or accuracy against the training dataset size.

Scikit Learn: `sklearn.model_selection.learning_curve`

### Error Analysis
* Filter on failed predictions and manually look for patterns to help you pivot on target, key attributes, and failure types and build histograms on error counts

* Common Errors:
 - Data Problems (many variants for the same word)
 - labeling errors
 - unbalanced classes
 - discriminating information is not captured in feature set

### Regularization
Overfitting is often caused by over-complex models. Regularization adds a _penalty_ for added model complexity.

regularized cost = cost + (lambda / 2) * penalty

Different Penalty Types:
L1 regularization (Lasso Regression): penalty = the sum of the absolute value of the coefficients (weights)
 - L1 feature: the weight of any important feature is reduced to zero and acts as a feature selection method.
 - note: The features must be scaled first!
 - Scikit Learn: `sklearn.linear_model.Lasso`

L2 regularization (Ridge Regression): penalty =  the sum of the coefficients (weights) squared. L2 is a more gradual approach.
 - Scikit Learn: `sklearn.linear_model.Ridge`

Elastic Net Regression is linear regression with both L1 and L2.
 - Scikit Learn: `sklearn.linear_model.ElasticNet`

### Hyperparameter Tuning

Questions to ask when choosing the optimal model:
* What learning rate/nodes/layers should I use for the neural network model?
* What is the minimum number of samples I should use at the leaf node in a decision tree or random forest model?
* What is the optimum C parameter to use in an SVM model?

Hyperparameter Optimization Techniques:
* Grid Search: allows you to search for the best parameter combination over a set parameters (very compute intensive!) Scikit Learn: `sklearn.grid_serach.GridSearchCV`
* Random Search: Each setting is sampled from a distribution over possible parameter values.

### Model Tuning

__Training Data Tuning__
* training set too small? Sample more data if possible
* Training Set biased against or missing some important scenarios? Sample and label more data for those scenarios if possible. For cases where more data is not available:
  - create synthetic data or duplicate data. Synthetic data is more flexible way to increase the sample size for the imbalanced classes.
  - SMOTE: Synthetic Oversampling/Understampling Technique

__Feature Set Tuning__
* Add features that help capture pattern for classes of errors
* Try different transformations of the same feature
* Apply dimensionality reduction to reduce impact of weak features

__Dimensionality Reduction__
Reduces the (effective) dimension of the data with minimal loss of information.

### Feature Extraction
Map data into a smaller feature space that captures the bulk of the information in the data (a.k.a. data compression)
* improves computational efficiency
* reduces curse of dimensionality

__Principal Component Analysis__
Unsupervised linear approach to feature selection. Finds patterns based on correlations between Features

Constructs principal components: orthogonal axes in directions of maximum variance.

Scikit Learn: `sklearn.decomposition.PCA`

Kernel PCA Example: If you have two features with a non-linear relationship, we can create a new feature that is the distance from the center. Using this new "distance feature" instead of the original two features, reduces the complexity without losing information.  

__Linear Discriminant Analysis__ (another method of dimensionality reduction)

### Feature Selection
The goal is to remove "noisy" or uninformative features
* Filter Methods: Measure the correlation between each independent variable and the response variable before training (less computationally expensive)
   - Pearson Correlation
   - Chi-Squared Test
   - ANNOVA test
   - Information Gain
* Wrapper Method: Used during training to measure the performance of each feature (computational expensive)
 - Genetic Algorithms
 - Backward feature elimination algorithms
 - Sequential feature selection algorithms
* Embedded Methods: Feature selection is embedded in the algorithm iteslf. (less computationally expensive than wrapper methods)
 - Lasso regression
 - Random Forest

### Bagging & Boosting Algorithms
Approaches to semi-automated feature selection:

__Bagging (a.k.a. Bootstap Aggregating)__
Generate a group of weak learners that when combined together generate higher accuracy. Subset sampling with replacement. Predictions are the aggregated of all the weak learners. _Is a good method for models with high variance_ - such as with decision trees. Bagging reduces the variance without changing the bias.

Scikit Learn: `sklearn.ensemble.BaggingClassifier` and `sklearn.ensemble.BaggingRegressor`

__Boosting__
Assigns "strengths" to each weak learner and iteratively trains learners using misclassified examples.It is good to use with models that have a high bias and can accept weights on individual samples. Available on Scikit Learn and the `XGBoost` Python library.

Scikit Learn:
 - `sklearn.ensemble.AdaBoostClassifier`
 - `sklearn.AdaBoostRegressor`
 - `sklearn.GradientBoostingClassifier`


## Model Evaluation & Productionizing ML

### Using ML in Production
Considerations:
 - model hosting
 - model deployment
 - pipelines to provide feature vectors
 - code to provide low-latency and/or high volume predictions
 - Model and data updating versioning
 - quality monitoring
 - data and model security encryption
 - customer privacy, fairness, and trust
 - data provider contractual constraints (e.g. attribution, cross-fertilization)

Types of Production Environments:
1. Batch predictions
 - useful if all possible inputs are known
 - predictions can still be served real-time, simply read from pre-computed values
2. Online predictions
 - useful if input space is Large
 - low latency requirement (at most 100ms)
3. Online training
 - sometimes data training patterns change often, so need to train online (e.g. fraud detection)

### Model Evaluation Metrics
Business metrics may not be the same as the performance metrics that are optimized during training. (e.g. click-through rate) Ideally, performance metrics are highly correlated with business metrics.

__The Confusion Matrix__
True Positives (TP) = all predictions that were correctly labeled as TRUE
True Negatives (TN) = all predictions that were correctly labeled as FALSE
False Positives (FP) = all prediction that were _incorrectly_ labeled as FALSE  
False Negatives (FN) = all predictions that were _incorrectly_ labeled as TRUE

__Accuracy__ = The set of all predictions that were correctly predicted divided by all predictions made (TP + TN) / (TP + TN + FP + FN)

When True Negatives (TN) are so large they dwarf the other categories then you may want to look at Precision or Recall instead of Accuracy.

__Precision__ = The ratio of observations correctly labeled as TRUE to all actual TRUE  

Precision = TP / (TP + FP)

__Recall__ = The ratio of observations correctly labeled as TRUE to all predicted TRUE

Recall = TP / (TP + FN)

__F1 Score__ = a combination of Precision and Recall (the harmonic mean)

F1 = (2 * Precision * Recall) / (Precision + Recall)

### Cross Validation
Using a portion of the total training set to train and evaluate in order to avoid overfitting the model.

Scikit Learn:
`train, test = sklearn.model_selection.train_test_split(df, test_size=0.3)`

__K-Fold Cross Validation__
issue: small dataset = small training set = not enough data for good training & leaves an unrepresentative test set leading to invalid metrics. The solution is to use all the data available.

1. randomly partition data into k "folds"
2. for each fold, train a model on other k-1 folds and evaluate on that fold
3. train a model using the whole dataset without risking invalidation.
4. Average the resulting metrics

Choose K:
--> Large K = more time, more variance
--> Small K = more bias
(k = 5 to k = 10 is typical)

__Stratified k-Fold__ preserves class proportions

__Leave-one-out k-Fold__ is used for very small datasets


### Metrics for Linear Regression
Problem: The confusion matrix doesn't make sense for Regression

Regression metrics:
* __MSE (Mean Squared Error):__ average the squared error over entire dataset (very commonly used)
  - Scikit Learn: `sklearn.metrics.mean_squared_error`
* __R-squared:__ The fractional variance accounted for by the model (percent from 0 to 1). R-squared is the proportion of the variance in the model that is explainable by the model. Adding more variables often increases R-squared but then increases the overall variance and can lead to overfitting.
* __Adjusted R-squared:__ Only allows R-squared to increase if the assed variables have a significant effect on the prediction.

### Using ML Models in Production: Storage
* Row Oriented formats
 - CSV file
 - read-only Database (RODB): internal read-only file with key-based access  
 - Avro: allows schema evolution for Hadoop
* Column Oriented formats
 - Parquet: type-aware and indexed for Hadoop
 - Optimized Row Columnar (ORC): Type aware, indexed, and with statistics for Hadoop
* User Defined Formats
 - JSON: key-value objects
 - HDF5: Hierarchical Data Format 5 - flexible data model with chunks
* Compression can be applied to all formats
* Usual trade-offs: Read/Write speeds, size, platform dependency, ability for schema to evolve, schema/data separability, type richness  

__Model & Pipeline Persistence__
Problems can occur when the production system does not support training of the final model because of incompatible platforms.

* Predictive Model Markup Language (PMML): vendor independent XML-based language for storing ML models
 - KNIME (analytics/ML library): full support
 - Scikit-learn: extensive support
 - Spark MLlib: limited support    

* Custom Methods:
 - Scikit-learn: uses python `Pickle` method to serialize/deserialize Python objects
 - Spark MLlib: Transformers and Estimators implement MLWritable
 - TensorFlow: Allows saving of MetaGraph
 - MxNet: saves into JSON

__Model Deployment__

technology Transfer: Experimental framework may not suffice for production. A/B testing or shadow testing helps catch production issues early.

### Using ML Models in Production: Monitoring & Maintenance
Its important to monitor quality metrics and business impacts with dashboards, alarms, user feedback, etc...
* The real-world domain may change over time
* the software environment may change
* High profile special cases may fail
* There may be a change in business goals

Performance deterioration may require new tuning:
- changing goals may require new metrics
- A changing domain may require changes to the validation set or changes in the features
- Your validation set may be replaced over time to avoid overfitting.

### Using ML Models in Production: AWS Ecosystem
* SageMaker - build, train, deploy at scale
* Rekognition - image analysis for picture and video
* Lex - speech recognition and chatbots
* Polly - text-to-voice with more than 2 dozen languages with a variety of voices
* Comprehend - NLP text classification
 - discover insights and relationships in text
 - identify the language based on the text
 - extract key phrases, places, people, brands, or events
 - sentiment analysis (understand how positive or negative the text is)
 - automatically organize a collection of text files by topic
* Translate - perform fluent translation of text and localize content for international users
* Transcribe
 - Automatic speech recognition for speech-to-text capability on your applications
 - analyze audio files and return a text document.
* AWS DeepLens - Image and video hardware device that is a tool to help software engineers develop business use-cases.
* AWS Glue: Data integration service for managing ETL jobs
* Deep Scalable Sparse Tensor Network Engine (DSSTNE): Neural Network Engine for training your own deep learning model from scratch with AWS's scalable framework

### Common Mistakes in ML Projects
* You solved the wrong Problem
* The data was flawed
* The solution didn't scale
* The final result doesn't match with the prototype's result
* It takes too long to fail 
* The solution was too complicated
* There were not enough allocated engineering resources to try out long-term science ideas
* There was a lack of a true collaboration
