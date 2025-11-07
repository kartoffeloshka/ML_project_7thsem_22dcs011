Clustering-Based Gene Expression Analysis for Disease Subtype Discovery

1. Project Overview

This repository contains the source code and experimental results for a comprehensive analysis of unsupervised clustering algorithms applied to high-dimensional gene expression data. The primary objective of this project is to evaluate the efficacy of different clustering methods in identifying and partitioning distinct disease subtypes based on genomic profiles.

The analysis framework is systematically applied to three different gene expression datasets, comparing the performance of four prominent clustering algorithms.

2. Clustering Algorithms Implemented

The following four unsupervised clustering methods are implemented and evaluated:

K-Means Clustering: A partitional clustering algorithm based on minimizing intra-cluster variance (inertia).

Agglomerative Hierarchical Clustering: A bottom-up clustering approach that progressively merges similar clusters, visualized via dendrograms.

Spectral Clustering: A graph-based algorithm that uses the eigenvalues (spectrum) of a similarity matrix to perform dimensionality reduction before clustering.

Gaussian Mixture Models (GMM): A probabilistic model that assumes data is generated from a mixture of a finite number of Gaussian distributions with unknown parameters.

3. Datasets

This study utilizes three distinct, publicly available gene expression datasets:

Brain Cancer Dataset (GSE50161): Contains gene expression profiles for multiple subtypes of brain tumors.

Breast Cancer Dataset (GSE45827): Contains gene expression profiles for various subtypes of breast cancer.

[Third Dataset Name - e.g., Leukemia (GSEXXXX)]: [Brief description of the third dataset].

For each dataset, the original "ground truth" subtype labels are retained for evaluation purposes but are not used during the clustering process.

4. Methodology

The analysis for each of the 12 experiments (4 algorithms x 3 datasets) follows a standardized pipeline:

Data Ingestion & Preparation: Loading the dataset and separating feature (gene expression) data from the ground truth labels.

Feature Scaling: Applying StandardScaler to normalize the high-dimensional feature space, ensuring all features have zero mean and unit variance.

Dimensionality Reduction (for Clustering): Utilizing Principal Component Analysis (PCA) to reduce the data to the number of components that capture 90% of the explained variance. This reduced-dimension dataset is used for model training to improve performance and computational efficiency.

Dimensionality Reduction (for Visualization): Applying PCA to reduce the scaled data to two principal components (PC1, PC2) for 2D scatter plot visualization.

Model Training: Fitting the respective clustering algorithm (K-Means, Hierarchical, Spectral, GMM) to the high-variance PCA data.

Cluster Evaluation: Assessing the quality of the resulting clusters using a comprehensive suite of seven evaluation metrics.

5. Evaluation Metrics

Cluster performance is quantified by comparing the algorithm's predicted labels against the known ground truth labels. The following metrics are employed:

External Metrics (Require Ground Truth):

Adjusted Rand Index (ARI)

Fowlkes-Mallows Score (FMS)

Homogeneity Score

Completeness Score

V-measure

Internal Metrics (No Ground Truth Required):

Silhouette Score

Davies-Bouldin Score

6. Repository Contents

This repository is structured to provide full reproducibility of the analysis.

/Source_Code/: Contains all 12 Python scripts (e.g., kmeans_brain.py, gmm_breast.py).

/Results/Plots/: Contains all generated visualizations, including:

Elbow method plots for K-Means.

Dendrograms for Hierarchical Clustering.

AIC/BIC plots for GMM.

2D PCA scatter plots comparing "Ground Truth" vs. "Predicted Clusters" for all 12 experiments.

/Results/Evaluation_Metrics/: Contains screenshots of the terminal output, documenting the quantitative evaluation metric scores for each experiment.

/Data/: (Optional) Contains the raw .csv files for the three datasets.

7. How to Run

Prerequisites

Ensure the following Python libraries are installed:

pip install pandas matplotlib seaborn scikit-learn


Execution

Navigate to the directory containing the source code.

Ensure the required dataset (e.g., Brain_GSE50161.csv) is present in the same directory or update the filepath variable within the script.

Execute the desired script via the terminal:

python kmeans_brain_cancer.py


The script will print all 7 evaluation metrics to the console and save the corresponding plots as .png files in the directory.
