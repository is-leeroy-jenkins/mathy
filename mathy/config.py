'''
  ******************************************************************************************
      Assembly:                mathy
      Filename:                config.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2022

      Last Modified By:        Terry D. Eppler
      Last Modified On:        05-01-2025
  ******************************************************************************************
  <copyright file="config.py" company="Terry D. Eppler">

	     config.py
	     Copyright ©  2022  Terry Eppler

     Permission is hereby granted, free of charge, to any person obtaining a copy
     of this software and associated documentation files (the “Software”),
     to deal in the Software without restriction,
     including without limitation the rights to use,
     copy, modify, merge, publish, distribute, sublicense,
     and/or sell copies of the Software,
     and to permit persons to whom the Software is furnished to do so,
     subject to the following conditions:

     The above copyright notice and this permission notice shall be included in all
     copies or substantial portions of the Software.

     THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
     INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
     FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
     IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
     ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
     DEALINGS IN THE SOFTWARE.

     You can contact me at:  terryeppler@gmail.com or eppler.terry@epa.gov

  </copyright>
  <summary>
    config.py
  </summary>
  ******************************************************************************************
'''
import os
import multiprocessing
from pathlib import Path

# -------------- APP-LEVEL UTILITIES -------------

def throw_if( name: str, value: object ) -> None:
	"""Input guard.
	
	Purpose:
	    Validates that a required argument contains a usable value before the surrounding workflow
	    continues. This guard centralizes early validation so provider wrappers and UI routines fail
	    with consistent, readable error messages.
	
	Args:
	    name (str): Name value used by the operation.
	    value (object): Value value used by the operation.
	
	Returns:
	    None: This function performs its work through side effects and does not return a value."""
	if value is None:
		raise ValueError( f'Argument "{name}" cannot be None.' )
	
	if isinstance( value, str ) and not value.strip( ):
		raise ValueError( f'Argument "{name}" cannot be empty.' )
	
def get_bool( name: str, default: bool = False ) -> bool:
	"""Read a Boolean environment variable.

	Purpose:
		Converts environment-variable text into a deterministic Boolean value. Missing
		variables return the caller-provided default. Values of ``1``, ``true``, ``yes``,
		``y``, and ``on`` are treated as ``True``; all other defined values are treated as
		``False``.

	Args:
		name (str): Environment variable name.
		default (bool): Default value used when the environment variable is not defined.

	Returns:
		Parsed Boolean value, or the original default value when parsing fails.
	"""
	try:
		throw_if( 'name', name )
		value = os.getenv( name )
		return default if value is None else value.strip( ).lower( ) in ('1', 'true', 'yes', 'y',
			'on')
	except Exception:
		return default

def get_int( name: str, default: int ) -> int:
	"""Read an integer environment variable.

	Purpose:
		Parses an optional environment variable as an integer while preserving a safe
		default when the variable is missing, empty, or invalid. This keeps module import
		safe even when deployment configuration is incomplete.

	Args:
		name (str): Environment variable name.
		default (int): Default integer value used when parsing is not possible.

	Returns:
		Parsed integer value or the supplied default value.
	"""
	try:
		throw_if( 'name', name )
		value = os.getenv( name )
		return default if value in (None, '') else int( str( value ).strip( ) )
	except Exception:
		return default

def get_float( name: str, default: float ) -> float:
	"""Read a floating-point environment variable.

	Purpose:
		Parses an optional environment variable as a float while preserving a safe default
		when the variable is missing, empty, or invalid. This helper supports numeric
		configuration without making module import dependent on perfect environment state.

	Args:
		name (str): Environment variable name.
		default (float): Default floating-point value used when parsing is not possible.

	Returns:
		Parsed floating-point value or the supplied default value.
	"""
	try:
		throw_if( 'name', name )
		value = os.getenv( name )
		return default if value in (None, '') else float( str( value ).strip( ) )
	except Exception:
		return default

def get_path( name: str, default: Path ) -> Path:
	"""Read a path environment variable.

	Purpose:
		Resolves optional filesystem configuration from the environment. Missing variables
		return the resolved default path, and invalid values fall back to the resolved
		default path rather than interrupting module import.

	Args:
		name (str): Environment variable name.
		default (Path): Default path used when the environment variable is not defined.

	Returns:
		Resolved path value or the resolved default path.
	"""
	try:
		throw_if( 'name', name )
		throw_if( 'default', default )
		value = os.getenv( name )
		return Path( value ).resolve( ) if value else default.resolve( )
	except Exception:
		return default.resolve( )

def get_text( name: str, default: str ) -> str:
	"""Read a text environment variable.

	Purpose:
		Returns an environment variable as text while preserving the supplied default when
		the variable is missing or empty. This keeps optional configuration centralized and
		stable for callers that import the module early in application startup.

	Args:
		name (str): Environment variable name.
		default (str): Default text value.

	Returns:
		Environment value or supplied default.
	"""
	try:
		throw_if( 'name', name )
		value = os.getenv( name )
		return default if value in (None, '') else str( value )
	except Exception:
		return default

# ------------- COMMON CONSTANTS ---------------------
BASE_DIR = os.path.dirname( os.path.abspath( __file__ ) )
ROOT_DIR = Path( __file__ ).resolve( ).parent
LOG_DIR: Path = get_path( 'LOG_DIR', ROOT_DIR / 'logging' )
LOG_PATH: str = get_text( 'LOG_PATH', str( LOG_DIR / 'Exceptions.db' ) )
LOG_FILE: str = get_text( 'LOG_FILE', 'Exceptions' )
LOGO = r'resources/mathy_logo.png'
FAVICON = r'resources/favicon.ico'
BLUE_DIVIDER = "<div style='height:1.5px;align:left;background:#0078FC;margin:30px 0px 30px 0px;'></div>"
DEFAULT_DATA = r'stores/excel/Combined Schedules.xlsx'
DB_PATH = r'stores/sqlite/Data.db'
LLM_PATH = ''
DEFAULT_CTX = 4096
CORES = multiprocessing.cpu_count( )
MODES = [ 'Data Profile', 'Descriptive Statistics',  'Inferential Statistics', 'Anomaly Detection',
	'Classification Models', 'Regression Models', 'Clustering Models',
	'Time-Series Models', 'Data Upload', 'CRUD Ops', 'Data Filter',
	'Data Aggregation', 'SQL Console' ]

ML_MODE = [ 'Data Profile', 'Descriptive Statistics',  'Inferential Statistics', 'Anomaly Detection',
	'Classification Models', 'Regression Models', 'Clustering Models',  'Time-Series Models' ]

DB_MODE = [ 'Data Upload', 'CRUD Ops', 'Data Filter',
	'Data Aggregation', 'SQL Console' ]

REPO_URL = r'https://is-leeroy-jenkins.github.io/Mathy/'

MODE = { 'Data Profile': '🏗️ Data Profiling',
       'Descriptive Statistics': '🕵️ Descriptive Statistics',
       'Inferential Statistics': '🧠 Inferential Statistics',
       'Anomaly Detection': '🛸 Anomaly Detection',
       'Classification Models': '📊 Classification Analysis',
       'Regression Models': '📉 Regression Analysis',
       'Clustering Models': '🕸️ Clustering Models',
       'Time-Series Models': '⏱️ Time-Series Models',
       'Data Upload': '📤 Upload Data',
       'CRUD Ops': '➕ CRUD Operations',
       'Data Filter': '🔍 Filter Data' ,
       'Data Aggregation': '🧮 Aggregate Data',
       'SQL Console': '💻 SQL Console' }

# ------------- DEFINITIONS ---------------------

CLASSIFICATION_MODELS = r'''A classification algorithm that makes its predictions based on a
		linear predictor function combining a set of weights with the feature vector.'''

DATA_CARDINALITY = r'''Data cardinality refers to the uniqueness of data values contained in a
			particular column (field) of a database, or the numerical relationship between two
			linked tables. It is categorized as high (many unique values, e.g., UserID) or low
			(many repeated values, e.g., Gender), and it directly impacts query performance,
			indexing strategies, and database design
'''

QQ_PLOT = r'''A Q-Q (Quantile-Quantile) plot is a graphical tool used to assess if a data set
follows a specific theoretical distribution (commonly normal) by plotting sample quantiles against
theoretical quantiles. Points falling along a straight 45-degree line indicate a strong match,
while deviations suggest differences in distribution, skewness, or outliers.
'''

DESCRIPTIVE_STATISTICS = r'''Descriptive statistics summarize and organize data features using
measures of central tendency (mean, median, mode), variability (range, standard deviation, variance),
and shape (skewness). They provide simple, actionable summaries of a sample's characteristics without
making inferences about a larger population. Key types include measures of distribution, central tendency, and dispersion
'''

INFERENTIAL_STATISTICS = r'''Inferential statistics allows researchers to draw conclusions, make predictions,
or generalize findings about a large population based on data analyzed from a smaller sample. It uses
probability theory and hypothesis testing to determine if patterns are significant or due to chance,
helping to make informed decisions despite data limitations.
'''

CORRELATION_ANALYSIS = r'''Correlation analysis is a statistical method used to measure the
strength and direction of the relationship between two variables, yielding a coefficient (\(r\))
between -1 and +1. It identifies patterns (positive/negative) but does not prove causation.
Common types include Pearson (linear), Spearman (monotonic), and Kendall, crucial for finance,
research, and data analysis
'''

CORRELATION_STRUCTURE = r'''Correlation structures define the pattern of dependence between observations
in a dataset, crucial for analyzing repeated measures or clustered data where observations within subjects
are correlated. Common types include compound symmetry (constant correlation), AR(1) (decaying correlation over time),
and unstructured (unique correlations)
'''

NORMALITY_TESTING = r'''A normality test determines if a data set is well-modeled by a normal
distribution, a key assumption for parametric tests like t-tests and ANOVA. It uses graphical
methods (Q-Q plots, histograms) or statistical tests (Shapiro-Wilk, Kolmogorov-Smirnov) to check
if data follows a bell-shaped curve. A non-significant result (\(p > 0.05\)) generally indicates
the data is normally distributed.
'''

SHAPIRO_WILK = r'''The Shapiro-Wilk test is a formal statistical method to determine if a data set
follows a normal distribution (bell-shaped curve), with 'H': Data is normally distributed. A p-value
< 0.5 indicates the data deviates significantly from normality. It is highly effective for small
sample sizes (< 50)
'''

ANOVA = r'''Analysis of Variance (ANOVA) is a statistical method used to compare the means of three
or more groups to determine if at least one group mean is significantly different from the others.
It evaluates the importance of1 or more factors by comparing the variance between groups to the
variance within groups using an F-statistic: (Between-group variance) divided by (Within-group variance).

ANOVA partitions the total variability of a dataset into two components: variance between sample
means and variance within each sample. If the variance between groups is significantly higher than
within-group variance, the means are likely different.
'''

CATEGORICAL_ASSOCIATION_TEST = r'''A categorical association test, primarily the
Chi-Square Test of Independence, determines if a significant relationship exists between two
categorical variables by comparing observed frequencies to expected frequencies in a contingency table.
It tests the null hypothesis that variables are independent (no association). Common methods include
Chi-Square for large samples, Fisher’s exact test for small samples, and McNemar's for paired data.
'''

PEARSON_COEFFICIENT = r'''linear relationship between two continuous variables, ranging from -1 to +1.
A value of +1 indicates a perfect positive linear relationship, -1 a perfect negative relationship,
and 0 no linear correlation.
'''

SPEARMAN_COEFFICIENT = r'''Spearman's rank correlation coefficient 'rho' or 's' is a non-parametric
measure that assesses the strength and direction of the monotonic relationship between two ranked or
continuous variables. Ranging from -1 to +1, it evaluates how well the relationship can be described
by a monotonic function, without requiring normally distributed data.
'''

ECDF = r'''The Empirical Cumulative Distribution Function (ECDF) is a step function that represents
the fraction of data points less than or equal to a specific value, providing an empirical estimate
of the underlying cumulative distribution. It is calculated by sorting  observations and increasing
the function by 1/n at each data point, with values ranging from 0 to 1.
'''

Z_SCORE = r'''Score flags observations whose values are a specified number of standard deviations
away from the mean. This method works best when the variable is roughly symmetric and not
dominated by extreme skew or heavy tails.
'''

MODIFIED_Z = r'''Modified Z-Score uses the median and median absolute deviation (MAD) instead of
the mean and standard deviation. It is more robust than the standard Z-Score when the data contain
skew, heavy tails, or existing outliers.
'''

IQR = r'''IQR Fence flags observations below Q1 - k×IQR or above Q3 + k×IQR, where IQR is the
interquartile range. This is a simple and robust rule for detecting unusually low or high values
without assuming normality.
'''

MAHALANOBIS = r'''Mahalanobis Distance detects multivariate outliers by measuring how far each
observation is from the center of the data while accounting for covariance between variables.
It is  useful when unusual combinations of values matter more than extreme values in a single column.
'''

ISOLATION_FOREST = r'''Isolation Forest is an ensemble method that isolates unusual observations
through random partitioning. Points that are easier to isolate are treated as anomalies. It works
well for  nonlinear and high-dimensional patterns and does not require the data to be normally distributed.
'''

LOF = r'''Local Outlier Factor (LOF) compares the local density of each observation to the density
of its nearest neighbors. Points that lie in much sparser neighborhoods than nearby points are
flagged as anomalies. It is useful for detecting local anomalies that may not look extreme globally.
'''

Z_THRESHOLD = r'''Sets the cutoff used by both Z-Score and Modified Z-Score. Larger values make the
 rule more conservative and reduce the number of observations flagged as anomalies.
'''

IQR_MULTIPLIER = r'''Sets the multiplier applied to the interquartile range when building the lower
and upper IQR fences. Larger multipliers widen the fence and make the rule less sensitive.
'''

LOF_K = r'''Sets the number of nearest neighbors used by Local Outlier Factor. Smaller values
emphasize very local structure, while larger values smooth the density comparison over a broader neighborhood.
'''

MIN_METHODS = r'''Controls the consensus threshold. A row must be flagged by at least this many
methods before it is included in the final anomaly table.
'''

ANALYSIS_SCALE = r'''When enabled, the selected variables are standardized for analysis only.
This puts variables on a comparable scale so that multivariate methods are less dominated by columns
with large numeric ranges. The underlying dataset is not changed.
'''

CORRELATION_HEATMAP = r'''A Correlation Heatmap is a 2D graphical representation of a correlation matrix
			that uses colors to visualize the strength and direction of relationships between multiple
			numerical variables. It helps identify patterns, multicollinearity, and key features, with values
			ranging from -1 (perfect negative) to 1 (perfect positive).
			
			Pearson - linear relationship between two continuous variables, ranging from -1 to +1.
			A value of +1 indicates a perfect positive linear relationship, -1 a perfect negative
			relationship, and 0 no linear correlation.
			
			Spearman - Spearman's rank correlation coefficient 'rho' or 's' is a non-parametric
			measure that assesses the strength and direction of the monotonic relationship between
			two ranked or continuous variables. Ranging from -1 to +1, it evaluates how well the
			relationship can be described by a monotonic function, without requiring
			normally distributed data.
'''

CONFUSION_MATRIX = r'''A confusion matrix is a performance evaluation table for machine learning
		classification models, summarizing correct and incorrect predictions against actual data.
		It displays counts of true positives (TP), true negatives (TN), false positives (FP), and
		false negatives (FN), mapping the model's accuracy, precision, and recall'''

PERCLASS_ACCURACY = r'''Per-class accuracy is a classification model evaluation metric that
		measures the proportion of correct predictions for a specific class out of all samples
		belonging to that class. '''

PREDICTION_CONFIDENCE = '''Prediction confidence intervals quantify the uncertainty of a model's
		prediction for a new data point, typically providing a range (e.g., 95% confidence) within
		which an individual future observation is expected to fall. While confidence intervals estimate
		the mean response, prediction intervals are always wider because they account for both model
		uncertainty and individual data variability.'''

ROC_CURVE = r'''A Receiver Operating Characteristic (ROC) curve is a graph visualizing the performance
		of a binary classification model across all classification thresholds. It plots the True Positive Rate (Sensitivity)
		on the y-axis against the False Positive Rate (FPR) on the x-axis. It helps select optimal thresholds
		and compares models using the Area Under the Curve (AUC).'''

# -----------Outliers

ISOLATION_FOREST = r'''The Isolation Forest ‘isolates’ observations by randomly selecting a feature and then
		randomly selecting a split value between the maximum and minimum values of
		the selected feature. Since recursive partitioning can be represented by a tree structure,
		the number of splittings required to isolate a sample is equivalent to the path
		length from the root node to the terminating node. This path length, averaged over a
		forest of such random trees, is a measure of normality and our decision function.
'''

ONE_CLASS = r'''Encapsulates One- Class Support Vector Machine for novelty detection on high-dimensional data.
		The estimator learns a boundary around normal samples and flags observations
		outside that boundary as anomalies.
'''

OUTLIER_FACTOR = r'''Local Outlier Factor for unsupervised or novelty-based outlier detection.
		Provides decision function, prediction, and scoring interfaces.
'''

ELLIIPTIC_SQUARE = r'''Encapsulates  Elliptic Envelope for multivariate Gaussian-based outlier detection.
		This method is based on Mahalanobis distances under an elliptical (normal) distribution.
'''

# ---------- Classifiers
PERCEPTRON_CLASSIFIER = r'''The perceptron is an algorithm for supervised learning of binary classifiers.
		A binary classifier is a function that can decide whether or not an input, represented by a
		vector of numbers, belongs to some specific class. It is a type of linear classifier, i.e.
		a classification algorithm that makes its predictions based on a linear predictor function
		combining a set of weights with the feature vector.'''

LEAST_SQUARES_CLASSIFIER = r'''Least Squares Regression fits a linear model with coefficients w = (w1, …, wp)
		to minimize the residual sum of squares between the observed targets
		in the dataset, and the targets predicted by the linear approximation.
'''

LOGISTIC_REGRESSION = r''''A machine learning algorithm used for binary classification
		(predicting one of two outcomes, e.g., yes/no) by modeling probabilities using a
		sigmoid function. It calculates the likelihood of an event occurring, making it ideal
		for spam detection, credit scoring, and medical diagnosis.
'''

RIDGE_CLASSIFIER = r''''A classifier that first converts binary targets to {-1, 1} and then treats the problem as a
		regression task, optimizing the same objective as above. The predicted class corresponds
		to the sign of the regressor’s prediction. For multiclass classification, the problem is
		treated as multi-output regression, and the predicted class corresponds to the output
		with the highest value.
'''

LASSO_CLASSIFIER = r''''(Least Absolute Shrinkage and Selection Operator) is a regression analysis
		method that performs both variable selection and regularization to enhance model prediction
		accuracy and interpretability. By applying an  penalty to the regression model, it shrinks
		less important feature coefficients to exactly zero, effectively removing them.
'''

GRADIENT_DESCENT_CLASSIFIER = r''''Linear classifiers (SVM, logistic regression, etc.) with
		Stochastic Gradient Descent (SGD) training.  This estimator implements regularized
		linear models with stochastic gradient descent learning:
		
		The gradient of the loss is estimated each sample at a time and the model is updated along
		the way with a decreasing strength schedule (aka learning rate). SGD allows minibatch
		(online/out-of-core) learning via the partial_fit method. For best results using the
		default learning rate schedule, the stores should have zero mean and unit variance.
'''

NEAREST_NEIGHBOR_CLASSFIER = r''''The principle behind the k-nearest neighbor methods is to find
		a predefined number of training samples closest in distance to the new point,
		and predict the label from these. The number of samples can be a user-defined constant
		(k-nearest neighbor rate), or vary based on the local density of points
		(radius-based neighbor rate).
		
		The distance can, in general, be any metric measure: standard Euclidean distance is the
		most common choice. Neighbors-based methods are known as non-generalizing
		machine rate methods, since they simply “remember” all of its training df
		(possibly transformed into a fast indexing structure such as a Ball Tree or KD Tree).
'''

DESICION_TREE_CLASSIFIER = r''''Decision Trees (DTs) are a non-parametric supervised learning method used for
		classification. The goal is to create a model that predicts the value of a
		target variable by learning simple decision rules inferred from the stores feature_names.

		A tree can be seen as a piecewise constant approximation. Decision trees learn from stores
		to approximate a sine curve with a set of if-then-else decision rules.
		The deeper the tree, the more complex the decision rules and the fitter the model.
'''

RANDOM_FOREST_CLASSIFIER = r''''In random forests, each tree in the ensemble is built from a sample
		drawn with replacement (i.e., a bootstrap sample) from the training set. Splitting each node
		during the construction of a tree, the best split is found either from all input
		feature_names or a random subset of size max_features. The injected randomness in forests
		yield decision trees with decoupled prediction errors. By taking an average of those predictions,
		errors can cancel out. Random forests achieve a reduced variance
		by combining diverse trees, sometimes at the cost of a slight increase in bias.
		The variance reduction is often significant hence yielding an overall better model.
'''

GRADIENT_BOOST_CLASSIFIER = r''''A Boost classifier is a meta-estimator that begins by fitting a classifier
		on the original dataset and then fits additional copies of the classifier on the
		same dataset but where the weights of incorrectly classified instances are
		adjusted such that subsequent classifiers focus more on difficult cases.
'''

ADAPTIVE_BOOST_CLASSIFIER = r''''A Boost classifier is a meta-estimator that begins by fitting a classifier
		on the original dataset and then fits additional copies of the classifier on the
		same dataset but where the weights of incorrectly classified instances are
		adjusted such that subsequent classifiers focus more on difficult cases.
'''

BAGGING_CLASSIFIER = r''''Bagging methods form a class of algorithms which build several instances of a black-box
		 estimator on random subsets of the original training set and then aggregate their
		 individual predictions to form a final prediction. These methods are used as a way
		 to reduce the variance of a base estimator (e.g., a decision tree), by introducing
		 randomization into its construction procedure and then making an ensemble out of it.
		 In many cases, bagging methods constitute a very simple way to improve with respect
		 to a single model, without making it necessary to adapt the underlying base algorithm.
		 As they provide a way to reduce overfitting, bagging methods work best with strong and
		 complex models (e.g., fully developed decision trees), in contrast with boosting methods
		 which usually work best with weak models (e.g., shallow decision trees).
'''

VOTING_CLASSFIER = r''''The Voting Model is to combine conceptually different machine rate
		classifiers and use a majority vote or the average predicted probabilities (soft vote)
		to predict the class target_names. Such a classifier can be useful for a set of equally
		well performing model in order to balance out their individual weaknesses.
'''

STACKING_CLASSIFIER = r''''Stack of estimators with a final classifier. Stacked generalization consists in stacking
		the output of individual estimator and use a classifier to compute the final prediction.
		Stacking allows to use the strength of each individual estimator by using their output
		as input of a final estimator. Note that estimators_ are fitted on the full X while
		final_estimator_ is trained using cross-validated predictions of the base
		estimators using cross_val_predict.
'''

SUPPORT_VECTOR_CLASSIFIER = r'''' Support Vector Classifier (SVC) is asupervised machine
		learning algorithm used primarily for classification, though it also handles regression.
		It works by finding an optimal "hyperplane"—a decision boundary—that maximizes the margin
		(distance) between different data classes, which improves prediction accuracy and
		generalization to new datais based on libsvm. The fit time scales at least quadratically
		with the number of samples  and may be impractical beyond tens of thousands of samples.
'''

MULTILAYER_PERCEPTRON_CLASSIFIER = r'''Model optimizes the squared error using LBFGS or
		stochastic gradient descent.

		Activation function for the hidden layers:
		- ‘identity’, no-op activation, useful to implement linear bottleneck, returns f(x) = x
		- ‘logistic’, the logistic sigmoid function, returns f(x) = 1 / (1 + exp(-x)).
		- ‘tanh’, the hyperbolic tan function, returns f(x) = tanh(x).
		- ‘relu’, the rectified linear unit function, returns f(x) = max(0, x)
'''

# --------- Regressions

LEAST_SQUARES_REGRESSION = r''''Least-angle regression (LARS) is a regression algorithm for high-dimensional stores.
	    LARS is similar to forward stepwise regression. At each step, it finds the feature most
	    correlated with the target. When there are multiple features having equal correlation,
	    instead of continuing along the same feature, it proceeds in a direction equiangular
	    between the features.
'''

RIDGE_REGRESSION = r'''Solves a regression model where the loss function is the linear least squares function and
	    alpha is given by the l2-norm. Also known as Ridge Regression
	    or Tikhonov alpha. This estimator has built-in support for
	    multi-variate regression (i.e., when y is a 2d-array of shape (n_samples, n_targets))
	
	    The complexity parameter  controls the amount of shrinkage: the larger the value of alpha,
	    the greater the amount of shrinkage and thus the coefficients become
	    more robust to collinearity.
	
	    The algorithm used to fit the model is coordinate descent. To avoid unnecessary memory
	    duplication the X argument of the fit method should be directly passed as a
	    Fortran-contiguous numpy array. Regularization improves the conditioning of the problem
	    and reduces the variance of the estimates. Larger values specify stronger alpha.
	    Alpha corresponds to 1 / (2C) in other linear models such as LogisticRegression or LinearSVC.
	    If an array is passed, penalties are assumed to be specific to the targets.
'''

LASSO_REGRESSION = r'''Least-angle regression (LARS) is a regression algorithm for high-dimensional stores.
	    LARS is similar to forward stepwise regression. At each step, it finds the feature most
	    correlated with the target. When there are multiple features having equal correlation,
	    instead of continuing along the same feature, it proceeds in a direction equiangular
	    between the features.
'''

ELASTICNET_REGRESSION = r'''ElasticNet is a linear regression model trained with both L1 and
		L2-norm regularization of the coefficients. This combination allows for learning a sparse
		model where few of the weights are non-zero like Lasso, while still maintaining the
		regularization properties of Ridge. We control the convex combination of and using the
		l1_ratio parameter.
	
	    Elastic-net is useful when there are multiple feature_names that are correlated with one another.
	    Lasso is likely to pick one of these at random, while elastic-net is likely to pick both.
'''

LEAST_ANGLE_REGRESSION = r'''Least-angle regression (LARS) is a regression algorithm for high-dimensional stores.
	    LARS is similar to forward stepwise regression. At each step, it finds the feature most
	    correlated with the target. When there are multiple features having equal correlation,
	    instead of continuing along the same feature, it proceeds in a direction equiangular
	    between the features.
'''

BAYESIAN_RIDGE_REGRESSION = r'''Bayesian regression techniques can be used to include alpha parameters in the
	    estimation procedure: the alpha parameter is not set in a hard sense
	    but tuned to the df at hand. This can be done by introducing uninformative priors over
	    the hyperparameters of the model. The alpha used in Ridge regression and
	    classification is equivalent to finding a maximum a posteriori estimation under a
	    Gaussian prior over the coefficients with precision. Instead of setting lambda manually,
	    it is possible to treat it as a random variable to be estimated
'''

GRADIENT_DESCENT_REGRESSION = r'''Stochastic Gradient Descent (SGD) is a simple yet very
		efficient approach to discriminative rate of linear classifiers under convex loss functions such as
	    (linear) Support VectorStore Machines and Logistic Regression. Even though SGD has been around
	    in the machine rate community for a long time, it has received a considerable amount
	    of attention just recently in the context of large-scale rate.
	
	    SGD has been successfully applied to large-scale and sparse machine rate problems
	    often encountered in text classification and natural language processing.
	    Given that the df is sparse, the classifiers in this module easily scale to problems
	    with more than 10^5 training examples and more than 10^5 feature_names.
	
	    The regularizer is a penalty added to the loss function that shrinks model parameters
	    towards the zero vector using either the squared euclidean norm L2 or the absolute norm L1
	    or a combination of both (Elastic Net). If the parameter update crosses the 0.0 value
	    because of the regularizer, the update is truncated to 0.0 to allow for learning sparse
	    models and achieve online feature selection.
'''

NEAREST_NEIGHBOR_REGRESSION = r'''The principle behind k-nearest neighbor methods is to find a predefined number of
	    training samples closest in distance to the new point, and predict the label from these.
	    The number of samples can be a user-defined constant (k-nearest neighbor rate),
	    or vary based on the local density of points (radius-based neighbor rate).
	    The distance can, in general, be any metric measure: standard Euclidean distance is the
	    most common choice. Neighbors-based methods are known as non-generalizing
	    machine rate methods, since they simply “remember” all of its training df
	    (possibly transformed into a fast indexing structure such as a Ball Tree or KD Tree).
'''

DESICION_TREE_REGRESSION = r'''Decision Trees (DTs) are a non-parametric supervised learning method used for
	    regression. The goal is to create a model that predicts the value of a
	    target variable by learning simple decision rules inferred from the stores feature_names.
	
	    A tree can be seen as a piecewise constant approximation. Decision trees learn from stores
	    to approximate a sine curve with a set of if-then-else decision rules.
	    The deeper the tree, the more complex the decision rules and the fitter the model.
'''

EXTRA_TREES_REGRESSION = r'''An ensemble learning method that fits multiple randomized decision trees
		on sub-samples of a dataset. It enhances predictive accuracy and controls over-fitting by
		using random split thresholds for features, rather than searching for the best possible split,
		resulting in faster computation times and lower variance compared to standard Random Forests.
'''

RANDOM_FOREST_REGRESSION = r'''In random forests, each tree in the ensemble is built from a sample
		drawn with replacement (i.e., a bootstrap sample) from the training set.
	
	    Furthermore, when splitting each node during the construction of a tree,
	    the best split is found either from all input feature_names or a random subset of
	    size max_features.
	
	    The purpose of these two sources of randomness is to decrease the variance
	    of the forest estimator. Individual decision trees typically exhibit high variance
	    and tend to overfit. The injected randomness in forests yield decision trees with
	    decoupled prediction errors. By taking an average of those predictions,
	    some errors can cancel out. Random forests achieve a reduced variance
	    by combining diverse trees, sometimes at the cost of a slight increase in bias.
	    The variance reduction is often significant hence yielding an overall better model.
'''

GRADIENT_BOOST_REGRESSION = r'''Gradient Boosting builds an additive model in a forward stage-wise fashion;
    it allows for the optimization  of arbitrary differentiable loss functions.
    In each stage n_classes_ regression trees are  fit on the negative gradient of the binomial
    or multinomial deviance loss function. Binary classification is a special case where
    only a single regression tree is induced.
'''

ADAPTIVE_BOOST_REGRESSION = r'''An AdaBoost [1] regressor is a meta-estimator
		that begins by fitting a regressor on the original dataset and then fits additional
		copies of the regressor on the same dataset but where the weights of instances are
		adjusted according to the error of the current prediction.
	
	    The core principle of Boost Regression is to fit a sequence of weak learners
	    (i.e., models that are only slightly better than random guessing,
	    such as small decision trees) on repeatedly modified versions of the df.
	    The predictions from all of them are then combined through a weighted
	    majority vote (or sum) to produce the final prediction.
'''

BAGGING_MODEL_REGRESSION = r'''Bagging methods form a class of algorithms which build several instances of a black-box
	     estimator on random subsets of the original training set and then aggregate their
	     individual predictions to form a final prediction. These methods are used as a way
	     to reduce the variance of a base estimator (e.g., a decision tree), by introducing
	     randomization into its construction procedure and then making an ensemble out of it.
	
	     Bagging methods constitute a very simple way to improve with respect
	     to a single model, without making it necessary to adapt the underlying base algorithm.
	     As they provide a way to reduce overfitting, bagging methods work best with strong and
	     complex models (e.g., fully developed decision trees), in contrast with boosting methods
	     which usually work best with weak models (e.g., shallow decision trees).
'''

VOTING_MODEL_REGRESSION = r'''Prediction voting regressor for unfitted estimators.
		A voting regressor is an ensemble  meta-estimator that fits several base regressors,
		each on the whole dataset. Then it averages the individual predictions to form a final prediction.
'''

STACKING_MODEL_REGRESSION = r'''Stack of estimators with a final regressor. Stacked generalization
		consists in stacking the output of individual estimator and use a regressor to compute the final prediction.
	    Stacking allows to use the strength of each individual estimator by using
	    their output as input of a final estimator. Note that estimators_ are fitted on the
	    full X while final_estimator_ is trained using cross-validated predictions of
	    the base estimators using cross_val_predict.
'''

SUPPORT_VECTOR_REGRESSION = r'''Support Vector Regression (SVR) is a powerful supervised learning
		algorithm that predicts continuous values by finding a hyperplane (line or surface) that
		best fits data within a defined error tolerance ( -insensitive tube). Unlike traditional
		regression, SVR focuses on minimizing errors within a threshold, using only key data points,
		known as support vectors.
'''

GAUSSIAN_PROCESS_REGRESSION = r''' Allows prediction without prior fitting (based on the GP prior)
		provides an additional method sample_y(X), which evaluates samples
		drawn from the GPR (prior or posterior) at given inputs
		exposes a method log_marginal_likelihood(theta), which can be used externally
		for other ways of selecting hyperparameters, e.g., via Markov chain Monte Carlo.
'''

MULTILAYER_PERCEPTRON_REGRESSION = r'''A Multilayer Perceptron (MLP) is a foundational feedforward
			artificial neural network consisting of at least three layers—input, hidden, and
			output—of fully connected nodes. It uses nonlinear activation functions to model complex,
			non-linear relationships, making it capable of solving non-linear classification
'''

# --------- Cluster

KMEANS = r'''The KMeans algorithm clusters stores by trying to separate samples in n groups of equal
		variance, minimizing a criterion known as the inertia or within-cluster sum-of-squares.
		This algorithm requires the number of clusters to be specified.
		It scales well to large number of samples and has been used across a
		large range of application areas in many different fields.

		The algorithm has three steps. The first step chooses the initial centroids,
		with the most basic method being to choose samples from the dataset. After initialization,
		K-means consists of looping between the two other steps. The first step assigns each sample
		to its nearest centroid. The second step creates new centroids by taking the mean value of
		all of the samples assigned to each previous centroid. The difference between the old and
		the new centroids are computed and the algorithm repeats these last two steps until this
		value is less than a threshold. In other words, it repeats until the centroids do not move
		significantly.
'''

DBSCAN = r'''The DBSCAN algorithm views clusters as areas of high density separated by areas of low
		density. Due to this rather generic view, clusters found by DBSCAN can be any shape,
		as opposed to k-means which assumes that clusters are convex shaped. The central component
		to the DBSCAN is the concept of core samples, which are samples that are in areas of high
		density.

		A cluster is therefore a set of core samples, each close to each other (measured
		by some distance measure) and a set of non-core samples that are close to a core sample
		(but are not themselves core samples). There are two parameters to the algorithm,
		min_samples and eps, which define formally what we mean when we say dense. Higher
		min_samples or lower eps indicate higher density necessary to form a cluster.
'''

AGGLOMERATIVE = r'''The Agglomerative Cluster object performs a hierarchical clustering using a
		bottom up approach: each observation starts in its own cluster, and clusters are
		successively merged together. The linkage criteria determines the metric used for the merge
		strategy:

		'Minimize' the sum of squared differences within all clusters. It is a
		variance-minimizing approach and in this sense is similar to the k-means objective
		function but tackled with an agglomerative hierarchical approach.

		'Maximum' or complete linkage minimizes the maximum distance between observations of
		pairs of clusters. Average linkage minimizes the average of the distances between all observations of
		pairs of clusters.

		'Single' linkage minimizes the distance between the closest observations of pairs of
		clusters. Agglomerative Cluster can also scale to large number of samples when it is used jointly
		with a connectivity matrix, but is computationally expensive when no connectivity
		constraints are added between samples: it considers at each step all the possible merges.
'''

SPECTRAL = r'''Spectral Cluster does a low-dimension embedding of the affinity matrix between samples,
		followed by a KMeans in the low dimensional space. It is especially efficient if the
		affinity matrix is sparse and the pyamg module is installed. SpectralCluster requires
		the number of clusters to be specified. It works well for a small number of clusters but
		is not advised when using many clusters.

		For two clusters, it solves a convex relaxation of the normalised cuts problem on the
		similarity graph: cutting the graph in two so that the weight of the edges cut is small
		compared to the weights of the edges inside each cluster. This criteria is especially
		interesting when working on images: graph vertices are pixels, and edges of the similarity
		graph are a function of the gradient of the image.
'''

MEAN_SHIFT = r'''Mean Shift clustering aims to discover blobs in a smooth density of samples.
		It is a centroid based algorithm, which works by updating candidates for centroids to be
		the mean of the points within a given region. These candidates are then filtered in a
		post-processing stage to eliminate near-duplicates to form the final set of centroids.

		The algorithm automatically sets the number of clusters, instead of relying on a parameter
		bandwidth, which dictates the size of the region to search through. This parameter can be
		set manually, but can be estimated using the provided estimate_bandwidth function, which
		is called if the bandwidth is not set.

		The algorithm is not highly scalable, as it requires multiple nearest neighbor searches
		during the execution of the algorithm. The algorithm is guaranteed to converge,
		however the algorithm will stop iterating when the change in centroids is small.
'''

AFFINITY_PROPAGATION = r'''Affinity Propagation creates clusters by sending messages between pairs of samples until
		convergence. A dataset is then described using a small number of exemplars, which are
		identified as those most representative of other samples. The messages sent between pairs
		represent the suitability for one sample to be the exemplar of the other, which is updated
		in response to the values from other pairs. This updating happens iteratively until
		convergence, at which point the final exemplars are chosen,
		and hence the final clustering is given.
'''

BIRCH = r'''The Birch builds a tree called the Clustering Feature Tree (CFT) for the given stores.
		The stores is essentially lossy compressed to a set of Clustering Feature nodes (CF Nodes).
		The CF Nodes have a number of subclusters called Clustering Feature subclusters
		(CF Subclusters) and these CF Subclusters located in the non-terminal
		CF Nodes can have CF Nodes as children.

		The BIRCH algorithm has two parameters, the threshold and the branching factor.
		The branching factor limits the number of subclusters in a node and the threshold limits
		the distance between the entering sample and the existing subclusters.

		This algorithm can be viewed as an instance or stores reduction method, since it reduces
		the input stores to a set of subclusters which are obtained directly from the leaves of the
		CFT. This reduced stores can be further processed by feeding it into a global clusterer.
		This global clusterer can be set by n_clusters. If n_clusters is set to None,
		the subclusters from the leaves are directly read off, otherwise a global clustering step
		target_names these subclusters into global clusters (target_names) and the samples are
		mapped to the global label of the nearest subcluster.
'''

OPTICS = r'''The OPTICS is a generalization of DBSCAN that relaxes the eps requirement from a single
		value to a value range. The key difference between DBSCAN and OPTICS is that the OPTICS
		algorithm builds a reachability graph, which assigns each sample both a reachability_
		distance, and a spot within the cluster ordering_ attribute; these two attributes are
		assigned when the model is fitted, and are used to determine cluster membership.

		If OPTICS is run with the default value of inf set for max_eps, then DBSCAN style
		cluster extraction can be performed repeatedly in linear time for any given eps value
		using the cluster_optics_dbscan method. Setting max_eps to a lower value will result
		in shorter run times, and can be thought of as the maximum neighborhood radius from
		each point to find other potential reachable points.
'''

# -------- Scalers

STANDARD_SCALER = r'''Standardize features by removing the mean and scaling to unit variance.
		The standard score of a sample x is calculated as: z = ( x - u ) / s,
		where u is the mean of the training samples or zero if with_mean=False,
		and s is the standard deviation of the training samples or one if
		with_std=False.
'''

MINMAX_SCALER = r'''Transform features by scaling each feature to a given range.
		This estimator scales and translates each feature individually such
		that it is in the given range on the training set, e.g. between zero
		and one. This transformation is often used as an alternative to zero
		mean, unit variance scaling.
'''

ROBUST_SCALER = r'''Remove the median and scale features according to the quantile range.
		By default, the quantile range is the interquartile range ( IQR ), which
		is the range between the 1st quartile ( 25th quantile ) and the 3rd
		quartile ( 75th quantile ). Centering and scaling happen independently on each feature by computing
		the relevant statistics on the samples in the training set. The median
		and interquartile range are then stored for use on later data during
		transformation. Robust scaling is useful when outliers would otherwise negatively affect
		mean- and variance-based scaling methods.
'''

NORMAL_SCALER = r''' Normalizes samples individually to unit norm. Each sample ( that is,
		each row of the feature matrix ) with at least one non-zero component
		is rescaled independently of the other samples so that its norm
		( l1, l2, or max ) equals one. This transformer can work with dense NumPy arrays and sparse matrices.
		Scaling inputs to unit norms is a common preprocessing step for text
		classification and clustering. For example, the dot product of two
		l2-normalized TF-IDF vectors is the cosine similarity between them.
'''

MAXABS_SCALER = r'''Scale each feature by its maximum absolute value. This estimator scales and
		transforms each feature individually such that the maximal absolute value of each feature
		in the training set will be 1.0. It does not shift or center the data, and therefore
		does not destroy sparsity. This scaler can also be applied to sparse CSR or CSC matrices.
		MaxAbsScaler does not reduce the effect of outliers; it only linearly
		scales them down.
'''

MINMAX_SCALER = r'''Transform features by scaling each feature to a given range.
		This estimator scales and translates each feature individually such
		that it is in the given range on the training set, e.g. between zero
		and one. This transformation is often used as an alternative to zero
		mean, unit variance scaling. Min-Max Scaler does not reduce the effect of outliers, but it linearly
		scales them down into a fixed range, where the largest occurring data
		point corresponds to the maximum value and the smallest one
		corresponds to the minimum value.
'''

# --------- Transformers

FEATURE_HASHER = r''''Convert symbolic feature names to a matrix using feature hashing. This estimator
		is stateless and is intended for large-scale or memory-constrained workflows.
'''

DICT_VECTORIZER = r''''Transform lists of feature-value mappings to vectors. String-valued features are
		expanded using one-of-K style encoding, while numeric values are passed through
		as numeric feature values.
'''

HASH_VECTORIZER = r'''Convert a collection of text to a matrix of token occurrences. It turns a
		collection of text into a scipy.sparse matrix holding token occurrence counts
		(or binary occurrence information), possibly normalized as token frequencies
		if norm=’l1’ or projected on the Euclidean unit sphere if norm=’l2’. This text vectorizer
		implementation uses the hashing trick to find the token string name to feature integer index mapping.
		This strategy has several advantages it is very low memory scalable to large datasets as
		there is no need to store a vocabulary dictionary in memory.
'''

COUNT_VECTORIZER = r'''Convert a collection of text to a matrix of token counts. This implementation
		produces a sparse representation of the counts using scipy.sparse.csr_matrix. If you do not
		provide an a-priori dictionary and you do not use an analyzer that does some kind of
		feature selection then the number of feature_names will be equal to the vocabulary
		size found by analyzing the stores.
'''

TDIDF_VECTORIZER = r'''Tf means term-frequency while tf-idf means term-frequency times inverse document-frequency.
		This is a common term-weighting scheme in information retrieval, that has also found good
		use in document classification. The goal of using tf-idf instead of the raw frequencies of
		occurrence of a token in a given document is to scale down the impact of tokens that occur
		very frequently in a given corpus and that are hence empirically less informative than
		feature_names that occur in a small fraction of the training corpus. The formula that is used to
		compute the tf-idf for a term t of a document d in a document set is tf-idf(t, d) = tf(t, d) * idf(t), and the idf
		is computed as idf(t) = log [ n / df(t) ] + 1 (if smooth_idf=False), where n is the total
		number of text in the document set and df(t) is the document frequency of t;
		the document frequency is the number of text in the document set that contain
		the term t. The effect of adding “1” to the idf in the equation above is that
		terms with zero idf, i.e., terms that occur in all text in a training set,
		will not be entirely ignored. (Note that the idf formula above differs from the
		standard textbook notation that defines the idf as idf(t) = log [ n / (df(t) + 1) ]).
'''

COLUMN_TRANSFORMER = r''''Applies transformers to columns of an array or pandas DataFrame.
		This estimator allows different columns or column subsets of the input to be transformed
		separately and the features generated by each transformer will be concatenated to form
		a single feature space. This is useful for heterogeneous or columnar data,
		to combine several feature extraction mechanisms or transformations
		into a single transformer.
'''

TDIDF_TRANSFORMER = r'''Tf means term-frequency while tf-idf means term-frequency times inverse document-frequency.
		This is a common term-weighting scheme in information retrieval, that has also found good
		use in document classification. The goal of using tf-idf instead of the raw frequencies of
		occurrence of a token in a given document is to scale down the impact of tokens that occur
		very frequently in a given corpus and that are hence empirically less informative than
		feature_names that occur in a small fraction of the training corpus. Transform a count matrix
		to a normalized tf or tf-idf representation.
'''

MULTILABEL_BINARIZER = r'''Transform between an iterable of iterables and the multilabel binary matrix format.
		Each row in the transformed output indicates the presence or absence of each class
		label for a given sample.
'''

LABEL_BINARIZER = r'''Binarize labels in a one-vs-all fashion. This wrapper fits on target labels and
		transforms them to a binary matrix representation. It also supports converting the
		binary representation back to the original labels.
'''

BINARIZER = r''''Binarize data (set feature values to 0 or 1) according to a threshold.
		Values greater than the threshold map to 1, while values less than or equal to the
		threshold map to 0. With the default threshold of 0, only positive values map to 1.
		Binarization is a common operation on text count data where the analyst can decide to only
		consider the presence or absence of a feature rather than a quantified number of
		occurrences for instance. It can also be used as a pre-processing step for estimators
		that consider boolean random variables (e.g. modelled using the Bernoulli distribution in a Bayesian setting).
'''

# ----------- Encoders

ONEHOT_ENCODER = r'''Encode categorical features as a one-hot numeric array. The input to this
		transformer should be an array-like of integers or strings denoting the values
		taken on by categorical features. The features are encoded using a one-hot
		(aka one-of-K or dummy) encoding scheme. By default, the encoder derives categories from the unique values in each
		feature. Alternatively, categories may be specified manually. This encoding is
		commonly used for feeding categorical data to scikit-learn estimators,
		especially linear models and support vector machines.
'''

ORDINAL_ENCODER = r'''Transform each categorical feature into a single integer-valued feature
		ranging from 0 to n_categories - 1. Although this representation is useful for some workflows, the encoded
		values may imply an ordering that does not exist in the original categories.
		As a result, ordinal encoding should be used with care when the source
		features are nominal rather than ordinal.
'''

LABEL_ENCODER = r'''Encode target labels with values between 0 and n_classes - 1.
		This transformer is intended for encoding a one-dimensional target vector,
		not a feature matrix.
'''

TARGET_ENCODER = r'''Encode categorical features using the target values associated with each category.
		Each category is encoded using a shrunk estimate of the target mean conditioned on
		the category value and the global target mean. For multiclass targets, encodings are based on one-vs-all conditional target
		probabilities, which produces n_features * n_classes encoded output features. Missing values are treated as their own category.
		Categories not seen during training are encoded with the learned global target mean.
'''

POLYNOMIAL_FEATURES = r'''Generate polynomial and interaction features from the input feature matrix.
		This transformer creates a new feature matrix consisting of all polynomial
		combinations of the input features with degree less than or equal to the
		specified degree. For example, if an input sample is two-dimensional and of
		the form [a, b], the degree-2 polynomial features are
		[1, a, b, a^2, ab, b^2].
'''

# ---------- Imputers

MEAN_IMPUTER = r'''Impute missing values by replacing them with the arithmetic mean of each
		feature column.
'''

NEAREST_IMPUTER = r'''The NearestNeighborImputer class provides imputation for filling in missing values using
		the k-Nearest Neighbors approach. By default, a euclidean distance metric that supports
		missing values, nan_euclidean_distances, is used to find the nearest neighbors.
		Each missing feature is imputed using values from n_neighbors nearest neighbors that have
		a value for the feature. The feature of the neighbors are averaged uniformly or weighted
		by distance to each neighbor. If a sample has more than one feature missing, then the neighbors for that sample can be
		different depending on the particular feature being imputed. When the number of available
		neighbors is less than n_neighbors and there are no defined distances to the training set,
		the training set average for that feature is used during imputation. If there is at least
		one neighbor with a defined distance, the weighted or unweighted average of the
		remaining neighbors will be used during imputation. If a feature is always missing in
		training, it is removed during transform.
'''

ITERATIVE_IMPUTER = r'''The Iterative Imputer models each feature with missing values as a function of
		other features, and uses that estimate for imputation. It does so in an iterated
		round-robin fashion: at each step, a feature column is designated as output y and the
		other feature columns are treated as inputs X. A regressor is fit on (X, y) for known y.
		Then, the regressor is used to predict the missing values of y. This is done for each
		feature in an iterative fashion, and then is repeated for max_iter imputation rounds.
		The results of the final imputation round are returned.
'''

SIMPLE_IMPUTER = r'''Impute missing values using sklearn's Simple Imputer  for common strategy-based
		replacement operations.
'''

NEAREST_NEIGHBOR_IMPUTER = r'''Nearest neighbor imputation (kNN) fills missing data by locating the
		 most similar, complete records (donors) to a record with missing values (recipient) based on
		 distance metrics like Euclidean distance.
'''

# ---------- Features

PCA = r'''PCA (Principal Component Analysis) is a dimensionality reduction technique and helps
to reduce the number of features in a dataset while keeping the most important information. It changes
complex datasets by transforming correlated features into a smaller set of uncorrelated components.
It removes redundancy, improves computational efficiency and makes data easier to visualize and analyze.

PCA uses linear algebra to transform data into new features called principal components. It finds
these by calculating eigenvectors (directions) and eigenvalues (importance) from the covariance matrix.
PCA selects the top components with the highest eigenvalues and projects the data onto them simplify the dataset.
'''

CCA = r'''Canonical Correlation Analysis (CCA) is a multivariate statistical method used to
identify and quantify the relationships between two sets of variables, and measured on the same
subjects. It finds linear combinations of variables—canonical variates—that are maximally correlated.

'''

VARIANCE_THRESHOLD = r'''Variance Threshold is a simple baseline approach to feature selection. It removes all
		feature_names whose variance doesn’t meet some threshold. By default, it removes all
		zero-variance feature_names, i.e. feature_names that have the same value in all samples.
'''

SELECT_BEST = r'''A univariate feature selection works by selecting the best features based on univariate
		statistical tests. Removes all but the 'k' highest scoring features
'''

SELECT_PERCENT = r'''A univariate feature selection works by selecting the best features based on univariate
		statistical tests. It can be seen as a preprocessing step to an estimator.
		Removes all but a user-specified highest scoring percentage (default - 10%) of features
'''

SBS = r'''Implements Sequential Backward Selection (SBS) using a supplied
			classification estimator and scoring function. The algorithm begins with
			the full feature set and greedily removes one feature at a time until the
			desired number of features remains.
'''

RFE = r'''Recursive Feature Elimination (RFE) Given an external estimator that assigns weights
		to features (e.g., the coefficients of a linear model), recursive feature elimination (RFE)
		is to select features by recursively considering smaller and smaller sets of features.
		
		First, the estimator is trained on the initial set of features and the importance of each
		feature is obtained either through a coef_ attribute or
		through a feature_importances_ attribute. Then, the least important features are pruned
		from current set of features. That procedure is recursively repeated on the pruned set
		until the desired number of features to select is eventually reached.
'''

# ----- Correlation Analysis

PAIR_THRESHOLD = '''A correlation analysis pair threshold is a limit used to find feature pairs with
		a pairwise correlation coefficient above a set value. It helps drop extra columns in data
		science or build connection networks. Common absolute value limits range from 0.70 to 0.90
		for removing multi-collinearity.
'''

# ------ Scatter Analysis

SCATTER_ANALYSIS = '''A scatter analysis uses a scatter plot to visually display and evaluate the
		relationship between two numerical variables. By plotting data points on an X and Y Cartesian
		coordinate system, this analysis helps you quickly identify trends, patterns, and
		correlations in your data
'''

PEARSON_P = '''A Pearson p-value evaluates the statistical significance of correlation coefficient (r).
		It tests the null hypothesis that the true linear correlation between two continuous variables is zero
'''

PEARSON_R = '''A statistical measure that calculates the strength and direction of a linear relationship
		between two continuous variables, producing a score from -1 to 1.
'''

SPEARMAN_R = '''Spearman's rank correlation coefficient measures the strength and direction of a
		monotonic relationship between two ranked or ordinal variables. It acts as the non-parametric
		alternative to the Pearson correlation, meaning it does not assume normal data distribution
		or a linear relationship
'''

# ------ Categorical Distributions

CATEGORICAL_DISTRIBUTION = '''The categorical distribution is the generalization of the Bernoulli
		distribution for a categorical random variable, i.e. for a discrete variable with more than
		two possible outcomes, such as the roll of a die.
'''



