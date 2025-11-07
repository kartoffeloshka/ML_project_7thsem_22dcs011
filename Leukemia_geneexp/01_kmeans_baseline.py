import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
    silhouette_score,
    fowlkes_mallows_score, # <-- ADDED
    davies_bouldin_score   # <-- ADDED
)
import warnings # <-- ADDED

# Suppress warnings from K-Means n_init
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn.cluster._kmeans') # <-- ADDED

def load_and_prepare_data(filepath='Leukemia_GSE9476.csv'):
    """
    Loads the leukemia dataset, separates features and labels,
    and scales the features.
    """
    try:
        data = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        print("Please make sure the file is in the same directory as the script.")
        return None, None, None

    print("Dataset loaded successfully.")
    
    # Drop the 'samples' column as it's just an identifier
    if 'samples' in data.columns:
        data = data.drop('samples', axis=1)

    # Separate the true labels (disease type) from the gene expression features
    true_labels = data['type']
    features = data.drop('type', axis=1)
    
    # Get the number of unique disease types (for comparison later)
    n_true_clusters = true_labels.nunique()
    print(f"Found {n_true_clusters} unique disease subtypes in the 'type' column.")
    print(f"Subtypes: {true_labels.unique().tolist()}")

    # --- Pre-processing ---
    # 1. Scale the data: This is crucial for both PCA and K-Means,
    # as they are sensitive to features with different scales.
    print("Standardizing gene expression data...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    return features_scaled, true_labels, n_true_clusters

def perform_pca(features_scaled, n_components=2):
    """
    Performs PCA for dimensionality reduction, primarily for visualization.
    """
    print(f"Performing PCA to reduce dimensions to {n_components} components...")
    pca = PCA(n_components=n_components)
    features_pca = pca.fit_transform(features_scaled)
    
    print(f"Explained variance by {n_components} components: {pca.explained_variance_ratio_.sum() * 100:.2f}%")
    
    # Create a DataFrame for easy plotting
    pca_df = pd.DataFrame(
        data=features_pca,
        columns=['PC1', 'PC2']
    )
    return pca_df

def find_optimal_k(features_scaled, max_k=10):
    """
    Uses the Elbow Method to suggest an optimal number of clusters (k).
    We will use the PCA-reduced data for faster computation,
    but for a final model, you might use higher-dim PCA or original scaled data.
    """
    print("Finding optimal 'k' using the Elbow Method...")
    inertias = []
    
    # We use the scaled features, not the 2-component PCA,
    # for a more accurate inertia calculation.
    # Let's first reduce dimensions to a reasonable number, e.g., 90% variance
    pca_high_dim = PCA(n_components=0.90)
    features_pca_high_dim = pca_high_dim.fit_transform(features_scaled)
    print(f"Reduced to {features_pca_high_dim.shape[1]} components capturing 90% variance for Elbow Method.")
    
    k_range = range(2, max_k + 1)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42) # <-- ADDED n_init=10
        kmeans.fit(features_pca_high_dim)
        inertias.append(kmeans.inertia_)
    
    # Plot the elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertias, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia (WCSS)')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('elbow_plot.png')
    print("Saved 'elbow_plot.png'. Check this plot to help choose 'k'.")
    # We will proceed with n_true_clusters, but this plot is for reference.

def perform_clustering(features_scaled, pca_df, true_labels, n_clusters):
    """
    Performs K-Means clustering and evaluates the results.
    """
    print(f"\nPerforming K-Means clustering with k={n_clusters}...")
    
    # --- Clustering ---
    # We cluster on the scaled, high-dimensional data.
    # Clustering on just 2 PCA components is usually bad for performance,
    # it's just for visualization.
    # Let's use the 90% variance PCA data we made earlier.
    pca_high_dim = PCA(n_components=0.90)
    features_to_cluster = pca_high_dim.fit_transform(features_scaled)

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    predicted_clusters = kmeans.fit_predict(features_to_cluster)
    
    # Add cluster results to our 2D PCA DataFrame for plotting
    pca_df['predicted_cluster'] = predicted_clusters
    pca_df['true_label'] = true_labels
    
    # --- Evaluation ---
    print("\n--- Clustering Evaluation Metrics (7 total) ---")
    
    # 1. Metrics that require true labels (Ground Truth)
    print("\nMetrics Based on Ground Truth Labels:")
    ari = adjusted_rand_score(true_labels, predicted_clusters)
    fms = fowlkes_mallows_score(true_labels, predicted_clusters) # <-- ADDED
    homogeneity = homogeneity_score(true_labels, predicted_clusters)
    completeness = completeness_score(true_labels, predicted_clusters)
    v_measure = v_measure_score(true_labels, predicted_clusters)
    
    print(f"1. Adjusted Rand Index (ARI):       {ari:.4f} (Good: ~1.0, Random: 0.0)")
    print(f"2. Fowlkes-Mallows Score (FMS):   {fms:.4f} (Good: ~1.0, Bad: 0.0)")
    print(f"3. Homogeneity:                   {homogeneity:.4f} (Good: ~1.0, clusters contain only one class)")
    print(f"4. Completeness:                  {completeness:.4f} (Good: ~1.0, all members of a class are in one cluster)")
    print(f"5. V-measure (Homogeneity & Comp.): {v_measure:.4f} (Good: ~1.0, harmonic mean of the two)")


    # 2. Metrics that do *not* require true labels (Internal Metrics)
    print("\nInternal Metrics (No Ground Truth Needed):")
    silhouette = silhouette_score(features_to_cluster, predicted_clusters)
    db_score = davies_bouldin_score(features_to_cluster, predicted_clusters) # <-- ADDED
    
    print(f"6. Silhouette Score:            {silhouette:.4f} (Good: +1, Bad: -1, Overlapping: 0)")
    print(f"7. Davies-Bouldin Score:        {db_score:.4f} (Good: 0.0, Bad: higher values)")
    
    return pca_df

def plot_results(pca_df):
    """
    Generates two scatter plots:
    1. PCA colored by true disease subtype.
    2. PCA colored by K-Means predicted cluster.
    """
    print("\nGenerating result plots...")
    plt.figure(figsize=(20, 9))
    
    # Plot 1: Ground Truth
    plt.subplot(1, 2, 1)
    sns.scatterplot(
        x='PC1', y='PC2',
        hue='true_label',
        palette=sns.color_palette("hsv", n_colors=pca_df['true_label'].nunique()),
        data=pca_df,
        legend='full',
        s=100,
        alpha=0.8
    )
    plt.title('Ground Truth: Actual Disease Subtypes (PCA)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True, linestyle='--', alpha=0.6)

    # Plot 2: K-Means Clusters
    plt.subplot(1, 2, 2)
    sns.scatterplot(
        x='PC1', y='PC2',
        hue='predicted_cluster',
        palette=sns.color_palette("hsv", n_colors=pca_df['predicted_cluster'].nunique()),
        data=pca_df,
        legend='full',
        s=100,
        alpha=0.8
    )
    plt.title(f'K-Means Clustering Results (k={pca_df["predicted_cluster"].nunique()}) (PCA)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.suptitle('Leukemia Subtype Clustering: Ground Truth vs. K-Means', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('clustering_results.png')
    print("Saved 'clustering_results.png'.")
    print("\nDone! Check the console output for metrics and the saved .png files for plots.")

def main():
    # 1. Load and Prepare
    features_scaled, true_labels, n_true_clusters = load_and_prepare_data('Leukemia_GSE9476.csv')
    
    if features_scaled is None:
        return

    # 2. Reduce dimensions (for visualization)
    pca_df = perform_pca(features_scaled, n_components=2)
    
    # 3. Find optimal k (and save plot)
    # We know the true 'k' is n_true_clusters, but we run this
    # to show the process.
    find_optimal_k(features_scaled, max_k=10)
    
    # 4. Perform Clustering
    # We will use the *known* number of clusters for our final model
    # to see how well K-Means can find them.
    pca_df_results = perform_clustering(features_scaled, pca_df, true_labels, n_true_clusters)
    
    # 5. Plot Results
    plot_results(pca_df_results)

if __name__ == "__main__":
    main()