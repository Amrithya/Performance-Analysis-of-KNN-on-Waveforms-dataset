# Performance-Analysis-of-KNN-on-Waveforms-dataset

KNN with KNeighborsClassifier
This Jupyter Notebook (knn_with_KNeighborsClassifier.ipynb) demonstrates the implementation of the K-Nearest Neighbors (KNN) algorithm using the KNeighborsClassifier from scikit-learn. The notebook covers various aspects of KNN, including data preprocessing, model training, hyperparameter tuning, and evaluation. Additionally, it explores techniques like data reduction, dimensionality reduction, and imbalanced learning.

Table of Contents
Introduction
Importing Libraries
Importing the Dataset
Reading and Describing the Dataset
Correlation Matrix
Dataset Split
Implementation of KNeighborsClassifier
Calculating Accuracy
Tuning of K
Methods to Speed Up Calculations
Data Reduction Technique
Dimensionality Reduction
Imbalanced Learning

Introduction <a name="introduction"></a>
K-Nearest Neighbors (KNN) is a simple and effective algorithm used for both classification and regression tasks. It belongs to the family of instance-based, lazy learning algorithms. This notebook focuses on classification using the KNeighborsClassifier from scikit-learn, covering various aspects such as model tuning, data reduction, dimensionality reduction, and handling imbalanced datasets.

Importing Libraries <a name="importing-libraries"></a>
The notebook starts by importing necessary libraries, including scikit-learn, imbalanced-learn, and matplotlib. These libraries are used for data manipulation, visualization, and implementing the KNN algorithm.

Importing the Dataset <a name="importing-the-dataset"></a>
The dataset is loaded from Google Drive using the Colab environment. The file is stored in the variable df, and the dataset's location can be modified based on user requirements.

Reading and Describing the Dataset <a name="reading-and-describing-the-dataset"></a>
This section provides an overview of the dataset's structure, including its shape and descriptive statistics. The dataset contains waveform data with three different classifications (0, 1, and 2).

Correlation Matrix <a name="correlation-matrix"></a>
A correlation matrix is calculated to understand the relationships between different features in the dataset.

Dataset Split <a name="dataset-split"></a>
The dataset is split into training and testing sets using the train_test_split function from scikit-learn.

Implementation of KNeighborsClassifier <a name="implementation-of-kneighborsclassifier"></a>
The KNeighborsClassifier is implemented, and the model is trained on the training set.

Calculating Accuracy <a name="calculating-accuracy"></a>
The accuracy of the model is calculated using the accuracy_score function from scikit-learn.

Tuning of K <a name="tuning-of-k"></a>
Cross-validation is performed with different values of k to find the optimal value for the number of neighbors in the KNN algorithm.

Methods to Speed Up Calculations <a name="methods-to-speed-up-calculations"></a>
Two methods, KDTree and BallTree, are explored to speed up nearest neighbors search calculations.

Data Reduction Technique <a name="data-reduction-technique"></a>
The Condensed Nearest Neighbors (CNN) algorithm is applied to reduce the dataset size while preserving its classification accuracy.

Dimensionality Reduction <a name="dimensionality-reduction"></a>
Feature scaling using StandardScaler and dimensionality reduction using Principal Component Analysis (PCA) are performed to reduce the dataset's dimensionality.

Imbalanced Learning <a name="imbalanced-learning"></a>
The notebook explores imbalanced learning using the Synthetic Minority Over-sampling Technique (SMOTE) to balance the class distribution. Accuracy and F1-score are compared for different values of k.

The results and visualizations generated during these analyses are saved as PNG images for reference.
