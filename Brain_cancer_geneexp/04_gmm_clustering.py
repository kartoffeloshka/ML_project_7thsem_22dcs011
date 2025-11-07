import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture 
from sklearn.metrics import (
    adjusted_rand_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
    silhouette_score,
    fowlkes_mallows_score,  
    davies_bouldin_score   
)
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn.cluster._kmeans')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.manifold._spectral_embedding')
warnings.filterwarnings('ignore', category=ConvergenceWarning, module='sklearn.mixture._base')


def load_and_prepare_data(filepath='Brain_GSE50161.csv'):
    try:
        data = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        print("************************************************************")
        print(f"*** Make sure your file is named '{filepath}'    ***")
        print(f"*** and is in the same directory as the script.      ***")
        print("************************************************************")
        return None, None, None, None

    print(f"Dataset '{filepath}' loaded successfully.")
    
    if 'sample' in data.columns:
        data = data.drop('sample', axis=1)

    if 'type' not in data.columns:
        print("Error: 'type' column not found.")
        print("Please ensure your label column is named 'type'.")
        return None, None, None, None
        
    true_labels = data['type']
    features = data.drop('type', axis=1)
    
    n_true_clusters = true_labels.nunique()
    print(f"Found {n_true_clusters} unique disease subtypes in the 'type' column.")
    print(f"Subtypes: {true_labels.unique().tolist()}")

    print("Standardizing gene expression data...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    print("Performing high-dim PCA (90% variance) for clustering...")
    pca_high_dim = PCA(n_components=0.90)
    features_to_cluster = pca_high_dim.fit_transform(features_scaled)
    print(f"Reduced to {features_to_cluster.shape[1]} components capturing 90% variance.")
    
    return features_scaled, features_to_cluster, true_labels, n_true_clusters

def perform_pca_visual(features_scaled, n_components=2):
    print(f"Performing PCA to reduce dimensions to {n_components} components for visualization...")
    pca = PCA(n_components=n_components)
    features_pca_2d = pca.fit_transform(features_scaled)
    
    print(f"Explained variance by {n_components} components: {pca.explained_variance_ratio_.sum() * 100:.2f}%")
    
    pca_df = pd.DataFrame(
        data=features_pca_2d,
        columns=['PC1', 'PC2']
    )
    return pca_df

def plot_gmm_metrics(features_to_cluster, max_k=10):
    print("Finding optimal 'k' using AIC/BIC...")
    k_range = range(2, max_k + 1)
    aics = []
    bics = []
    
    for k in k_range:
        gmm = GaussianMixture(n_components=k, random_state=42, n_init=5)
        gmm.fit(features_to_cluster)
        aics.append(gmm.aic(features_to_cluster))
        bics.append(gmm.bic(features_to_cluster))
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, aics, 'bo-', label='AIC (Akaike Information Criterion)')
    plt.plot(k_range, bics, 'ro-', label='BIC (Bayesian Information Criterion)')
    plt.title('GMM Model Selection (Brain Tumor)')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Information Criterion Value')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('brain_tumor_gmm_aic_bic.png')
    print("Saved 'brain_tumor_gmm_aic_bic.png'. The 'elbow' (minimum) suggests the best k.")

def perform_gmm_clustering(features_to_cluster, pca_df_2d, true_labels, n_clusters):
    print(f"\nPerforming GMM clustering with k={n_clusters}...")
    
    gmm_model = GaussianMixture(
        n_components=n_clusters, 
        random_state=42,
        n_init=10  
    )
    predicted_clusters = gmm_model.fit_predict(features_to_cluster)
    
    pca_df_2d['predicted_cluster'] = predicted_clusters
    pca_df_2d['true_label'] = true_labels
    
    print("\n--- Clustering Evaluation Metrics (7 total) ---")
    
    print("\nMetrics Based on Ground Truth Labels:")
    ari = adjusted_rand_score(true_labels, predicted_clusters)
    fms = fowlkes_mallows_score(true_labels, predicted_clusters)
    homogeneity = homogeneity_score(true_labels, predicted_clusters)
    completeness = completeness_score(true_labels, predicted_clusters)
    v_measure = v_measure_score(true_labels, predicted_clusters)
    
    print(f"1. Adjusted Rand Index (ARI):     {ari:.4f} (Good: ~1.0, Random: 0.0)")
    print(f"2. Fowlkes-Mallows Score (FMS):   {fms:.4f} (Good: ~1.0, Bad: 0.0)")
    print(f"3. Homogeneity:                   {homogeneity:.4f} (Good: ~1.0, clusters contain only one class)")
    print(f"4. Completeness:                  {completeness:.4f} (Good: ~1.0, all members of a class are in one cluster)")
    print(f"5. V-measure (Homogeneity & Comp.): {v_measure:.4f} (Good: ~1.0, harmonic mean of the two)")

    print("\nInternal Metrics (No Ground Truth Needed):")
    silhouette = silhouette_score(features_to_cluster, predicted_clusters)
    db_score = davies_bouldin_score(features_to_cluster, predicted_clusters)
    
    print(f"6. Silhouette Score:              {silhouette:.4f} (Good: +1, Bad: -1, Overlapping: 0)")
    print(f"7. Davies-Bouldin Score:          {db_score:.4f} (Good: 0.0, Bad: higher values)")
    
    return pca_df_2d

def plot_results(pca_df):
    print("\nGenerating result plots...")
    n_colors_true = pca_df['true_label'].nunique()
    n_colors_pred = pca_df['predicted_cluster'].nunique()
    
    plt.figure(figsize=(20, 9))
    
    plt.subplot(1, 2, 1)
    sns.scatterplot(
        x='PC1', y='PC2',
        hue='true_label',
        palette=sns.color_palette("hsv", n_colors=n_colors_true),
        data=pca_df,
        legend='full',
        s=100,
        alpha=0.8
    )
    plt.title('Ground Truth: Actual Disease Subtypes (PCA)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.subplot(1, 2, 2)
    sns.scatterplot(
        x='PC1', y='PC2',
        hue='predicted_cluster',
        palette=sns.color_palette("hsv", n_colors=n_colors_pred),
        data=pca_df,
        legend='full',
        s=100,
        alpha=0.8
    )
    plt.title(f'GMM Clustering Results (k={n_colors_pred}) (PCA)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.suptitle('Brain Tumor Subtype Clustering: Ground Truth vs. GMM', fontsize=16, y=1.02)
    
    plt.tight_layout()
    plt.savefig('brain_tumor_gmm_results.png')
    print("Saved 'brain_tumor_gmm_results.png'.")
    print("\nDone! Check the console output for metrics and the saved .png files for plots.")

def main():
    filename = 'Brain_GSE50161.csv' 
    
    features_scaled, features_to_cluster, true_labels, n_true_clusters = load_and_prepare_data(filename)
    
    if features_scaled is None:
        return

    pca_df_2d = perform_pca_visual(features_scaled, n_components=2)
    
    plot_gmm_metrics(features_to_cluster, max_k=10)
    
    pca_df_results = perform_gmm_clustering(features_to_cluster, pca_df_2d, true_labels, n_true_clusters)
    
    plot_results(pca_df_results)
    
    plt.show()

if __name__ == "__main__":
    main()