import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# --- MODIFIED IMPORTS ---
from sklearn.cluster import AgglomerativeClustering  # <-- Changed from KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
    silhouette_score,
    fowlkes_mallows_score,  # <-- Added per your other scripts
    davies_bouldin_score   # <-- Added per your other scripts
)
import warnings
from scipy.cluster.hierarchy import dendrogram, linkage  # <-- Added imports
# --- END MODIFIED IMPORTS ---

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn.cluster._kmeans')

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

# --- NEW FUNCTION (Replaces find_optimal_k) ---
def plot_dendrogram(features_to_cluster, n_true_clusters):
    """
    Generates a dendrogram to visualize the hierarchical clustering structure.
    This replaces the "Elbow Method" from the K-Means script.
    """
    print("Generating dendrogram...")
    
    # 'ward' linkage minimizes the variance of the clusters being merged.
    linked = linkage(features_to_cluster, method='ward')
    
    plt.figure(figsize=(14, 8))
    dendrogram(
        linked,
        orientation='top',
        truncate_mode='lastp',  # Show only the last 'p' merged clusters
        p=30,                   # 'p' = 30
        show_leaf_counts=True,
        leaf_rotation=90.,
        leaf_font_size=12.,
        show_contracted=True,
    )
    
    # Add a horizontal line to show the cut-off for the true number of clusters
    plt.axhline(y=plt.gca().get_ylim()[1]*0.6, color='r', linestyle='--', label=f'Cut for k={n_true_clusters}')
    
    plt.title('Hierarchical Clustering Dendrogram (Leukemia)')
    plt.xlabel('Patient Samples or Clusters (Truncated)')
    plt.ylabel('Distance (Ward Linkage)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6, axis='y')
    plt.tight_layout()
    plt.savefig('leukemia_dendrogram.png')
    print("Saved 'leukemia_dendrogram.png'. Check this plot to help choose 'k'.")
# --- END NEW FUNCTION ---


# --- MODIFIED FUNCTION (Replaces perform_clustering) ---
def perform_hierarchical_clustering(features_to_cluster, pca_df_2d, true_labels, n_clusters):
    """
    Performs Hierarchical (Agglomerative) clustering and evaluates the results.
    """
    print(f"\nPerforming Hierarchical clustering with k={n_clusters}...")
    
    # --- Clustering ---
    # We cluster on the high-dimension PCA data.
    hierarchical_model = AgglomerativeClustering(
        n_clusters=n_clusters, 
        linkage='ward'
    )
    predicted_clusters = hierarchical_model.fit_predict(features_to_cluster)
    
    # Add cluster results to our 2D PCA DataFrame for plotting
    pca_df_2d['predicted_cluster'] = predicted_clusters
    pca_df_2d['true_label'] = true_labels
    
    # --- Evaluation (Upgraded to 7 metrics) ---
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
    2. PCA colored by Hierarchical predicted cluster.
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

    # Plot 2: Hierarchical Clusters
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
    plt.title(f'Hierarchical Clustering Results (k={n_colors_pred}) (PCA)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.suptitle('Leukemia Subtype Clustering: Ground Truth vs. Hierarchical', fontsize=16, y=1.02)
    # --- END MODIFIED TITLE ---
    
    plt.tight_layout()
    # --- MODIFIED FILENAME ---
    plt.savefig('leukemia_hierarchical_results.png')
    print("Saved 'leukemia_hierarchical_results.png'.")
    # --- END MODIFIED FILENAME ---
    print("\nDone! Check the console output for metrics and the saved .png files for plots.")

def main():
    # 1. Load and Prepare
    filename = 'Leukemia_GSE9476.csv' 
    
    # --- MODIFIED RETURN VALUES ---
    features_scaled, features_to_cluster, true_labels, n_true_clusters = load_and_prepare_data(filename)
    # --- END MODIFIED RETURN VALUES ---
    
    if features_scaled is None:
        return

    # 2. Reduce dimensions (for visualization)
    pca_df_2d = perform_pca_visual(features_scaled, n_components=2)
    
    # --- MODIFIED FUNCTION CALLS ---
    # 3. Plot Dendrogram (replaces Elbow Plot)
    plot_dendrogram(features_to_cluster, n_true_clusters)
    
    # 4. Perform Clustering
    # We will use the *known* number of clusters for our final model
    # to see how well Hierarchical clustering can find them.
    pca_df_results = perform_hierarchical_clustering(features_to_cluster, pca_df_2d, true_labels, n_true_clusters)
    # --- END MODIFIED FUNCTION CALLS ---
    
    # 5. Plot Results
    plot_results(pca_df_results)
    
    # Show all plots at the end
    plt.show()

if __name__ == "__main__":
    main()