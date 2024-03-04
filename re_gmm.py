import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

def load_and_preprocess_data(filepath):
    """load and preprocess data from a CSV file."""
    try:
        data = pd.read_csv(filepath)
        print("data loaded successfully.")
    except FileNotFoundError:
        print(f"file {filepath} not found.")
        return None
    except Exception as e:
        print(f"an error occurred: {e}")
        return None
    
    for column in ['Square Footage', 'Bedrooms', 'Bathrooms', 'Price']:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors='coerce')
        else:
            print(f"column {column} missing in data.")
            return None

    data.dropna(inplace=True)
    print("data preprocessing completed.")
    return data

def standardize_features(data, feature_columns):
    """standardize features to have 0 mean and unit variance."""
    scaler = StandardScaler()
    features = scaler.fit_transform(data[feature_columns])
    return features

def perform_gmm_clustering(features, n_components=12):
    """perform gaussian mixture"""
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=0)
    gmm.fit(features)
    labels = gmm.predict(features)
    print("GMM clustering completed.")
    return labels

def calculate_cluster_descriptors(features, labels):
    """calc mean feature values for each"""
    descriptors = {}
    for cluster in np.unique(labels):
        cluster_data = features[labels == cluster]
        descriptors[cluster] = {
            'Avg Square Footage': np.mean(cluster_data[:, 0]),
            'Avg Bedrooms': np.mean(cluster_data[:, 1]),
            'Avg Bathrooms': np.mean(cluster_data[:, 2]),
            'Avg Price': np.mean(cluster_data[:, 3])
        }
    print("descriptors calculated")
    return descriptors

def add_cluster_labels_to_data(data, labels, descriptors):
    """append labels to data."""
    label_str = {cluster: f"SF {int(desc['Avg Square Footage'])} | BD {int(desc['Avg Bedrooms'])} | BA {int(desc['Avg Bathrooms'])} | PR ${int(desc['Avg Price'])}"
                 for cluster, desc in descriptors.items()}
    data['Cluster Label'] = [label_str[label] for label in labels]
    print("Cluster labels added to data.")
    return data

def process_and_output_data(filepath, n_components_gmm=12):
    """process  data from CSV, output data object with labels."""
    data = load_and_preprocess_data(filepath)
    if data is None:
        return None

    features = standardize_features(data, ['Square Footage', 'Bedrooms', 'Bathrooms', 'Price'])
    cluster_labels = perform_gmm_clustering(features, n_components=n_components_gmm)
    cluster_descriptors = calculate_cluster_descriptors(features, cluster_labels)
    processed_data = add_cluster_labels_to_data(data, cluster_labels, cluster_descriptors)
    
    output_filename = "processed_data.csv"
    processed_data.to_csv(output_filename, index=False)
    print(f"processed data saved to {output_filename}.")
    
    return processed_data

if __name__ == "__main__":
    processed_data = process_and_output_data("sample_home_data.csv")
    if processed_data is not None:
        print("data processing complete, a preview:")
        print(processed_data.head())
    else:
        print("processing failed.")
