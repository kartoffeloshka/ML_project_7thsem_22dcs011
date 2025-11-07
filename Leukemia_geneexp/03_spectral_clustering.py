import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# --- MODIFIED IMPORTS ---
from sklearn.cluster import SpectralClustering  # <-- Changed from AgglomerativeClustering
from sklearn.metrics import (
    adjusted_rand_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
    silhouette_score,
    fowlkes_mallows_score,  # <-- All 7 metrics included
    davies_bouldin_score   
)
import warnings
# --- REMOVED IMPORTS ---
# from scipy.cluster.hierarchy import dendrogram, linkage (Not needed for Spectral)
# --- END MODIFIED IMPORTS ---

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn.cluster._kmeans')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.manifold._spectral_embedding')


def load_and_prepare_data(filepath='Leukemia_GSE9476.csv'):
    """
    Loads the leukemia dataset, separates features and labels,
    and scales the features.
    """
    try:
        data = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        print("************************************************************")
        print(f"*** Make sure your file is named '{filepath}'     ***")
        print(f"*** and is in the same directory as the script.      ***")
        print("************************************************************")
        return None, None, None, None

    print(f"Dataset '{filepath}' loaded successfully.")
    
    # Drop the 'samples' column as it's just an identifier
    if 'samples' in data.columns:
        data = data.drop('samples', axis=1)

    # Separate the true labels (disease type) from the gene expression features
    if 'type' not in data.columns:
        print("Error: 'type' column not found.")
        print("Please ensure your label column is named 'type'.")
        return None, None, None, None
        
    true_labels = data['type']
    features = data.drop('type', axis=1)
    
    # Get the number of unique disease types (for comparison later)
    n_true_clusters = true_labels.nunique()
    print(f"Found {n_true_clusters} unique disease subtypes in the 'type' column.")
    print(f"Subtypes: {true_labels.unique().tolist()}")

    # --- Pre-processing ---
    # 1. Scale the data
    print("Standardizing gene expression data...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # 2. Create high-dim PCA data for clustering
    # This is more robust than clustering on the raw 20,000+ genes
    print("Performing high-dim PCA (90% variance) for clustering...")
    pca_high_dim = PCA(n_components=0.90)
    features_to_cluster = pca_high_dim.fit_transform(features_scaled)
    print(f"Reduced to {features_to_cluster.shape[1]} components capturing 90% variance.")
    
    return features_scaled, features_to_cluster, true_labels, n_true_clusters

def perform_pca_visual(features_scaled, n_components=2):
    """
    Performs PCA for 2D visualization.
    """
    print(f"Performing PCA to reduce dimensions to {n_components} components for visualization...")
    pca = PCA(n_components=n_components)
    features_pca_2d = pca.fit_transform(features_scaled)
    
    print(f"Explained variance by {n_components} components: {pca.explained_variance_ratio_.sum() * 100:.2f}%")
    
    # Create a DataFrame for easy plotting
    pca_df = pd.DataFrame(
        data=features_pca_2d,
        columns=['PC1', 'PC2']
    )
    return pca_df

# --- FUNCTION REMOVED ---
# plot_dendrogram() was specific to hierarchical and has been removed.
# Spectral clustering doesn't have a simple k-finding plot like Elbow or Dendrogram.
# --- END FUNCTION REMOVED ---


# --- MODIFIED FUNCTION (Replaces perform_hierarchical_clustering) ---
def perform_spectral_clustering(features_to_cluster, pca_df_2d, true_labels, n_clusters):
    """
    Performs Spectral clustering and evaluates the results.
    """
    print(f"\nPerforming Spectral clustering with k={n_clusters}...")
    
    # --- Clustering ---
    # We cluster on the high-dimension PCA data.
    # 'affinity="nearest_neighbors"' builds a graph based on local neighborhoods,
    # which is excellent for high-dimensional data.
    spectral_model = SpectralClustering(
        n_clusters=n_clusters, 
        affinity='nearest_neighbors',
        random_state=42,
        n_init=10  # Number of times to run with different seeds
    )
    predicted_clusters = spectral_model.fit_predict(features_to_cluster)
    
    # Add cluster results to our 2D PCA DataFrame for plotting
    pca_df_2d['predicted_cluster'] = predicted_clusters
    pca_df_2d['true_label'] = true_labels
    
    # --- Evaluation (All 7 metrics) ---
    print("\n--- Clustering Evaluation Metrics (7 total) ---")
    
    # 1. Metrics that require true labels (Ground Truth)
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


    # 2. Metrics that do *not* require true labels (Internal Metrics)
    print("\nInternal Metrics (No Ground Truth Needed):")
    silhouette = silhouette_score(features_to_cluster, predicted_clusters)
    db_score = davies_bouldin_score(features_to_cluster, predicted_clusters)
    
    print(f"6. Silhouette Score:              {silhouette:.4f} (Good: +1, Bad: -1, Overlapping: 0)")
    print(f"7. Davies-Bouldin Score:          {db_score:.4f} (Good: 0.0, Bad: higher values)")
    
    return pca_df_2d
# --- END MODIFIED FUNCTION ---

def plot_results(pca_df):
    """
    Generates two scatter plots:
    1. PCA colored by true disease subtype.
    2. PCA colored by Spectral predicted cluster.
    """
    print("\nGenerating result plots...")
    n_colors_true = pca_df['true_label'].nunique()
    n_colors_pred = pca_df['predicted_cluster'].nunique()
    
    plt.figure(figsize=(20, 9))
    
    # Plot 1: Ground Truth
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

    # Plot 2: Spectral Clusters
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
    # --- MODIFIED TITLE ---
    plt.title(f'Spectral Clustering Results (k={n_colors_pred}) (PCA)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.suptitle('Leukemia Subtype Clustering: Ground Truth vs. Spectral', fontsize=16, y=1.02)
    # --- END MODIFIED TITLE ---
    
    plt.tight_layout()
    # --- MODIFIED FILENAME ---
    plt.savefig('leukemia_spectral_results.png')
    print("Saved 'leukemia_spectral_results.png'.")
    # --- END MODIFIED FILENAME ---
    print("\nDone! Check the console output for metrics and the saved .png files for plots.")

def main():
    # 1. Load and Prepare
    filename = 'Leukemia_GSE9476.csv' 
    
    features_scaled, features_to_cluster, true_labels, n_true_clusters = load_and_prepare_data(filename)
    
    if features_scaled is None:
        return

    # 2. Reduce dimensions (for visualization)
    pca_df_2d = perform_pca_visual(features_scaled, n_components=2)
    
    # --- MODIFIED FUNCTION CALLS ---
    # 3. Perform Clustering
    # We will use the *known* number of clusters for our final model
    # to see how well Spectral clustering can find them.
    pca_df_results = perform_spectral_clustering(features_to_cluster, pca_df_2d, true_labels, n_true_clusters)
    # --- END MODIFIED FUNCTION CALLS ---
    
    # 4. Plot Results
    plot_results(pca_df_results)
    
    # Show all plots at the end
    plt.show()

if __name__ == "__main__":
    main()