import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from scipy.spatial.distance import euclidean
from textstat import flesch_reading_ease, flesch_kincaid_grade
from sklearn.cluster import KMeans
import seaborn as sns
from pathlib import Path
from utils.config import RESULTS_DIR

nltk.download('punkt')

def load_texts(original_dir, generated_file):
    """
    Load original and AI-generated texts from preprocessed directory.
    
    Args:
        original_dir: Path to directory containing preprocessed text files
        generated_file: Path to JSON file containing generated texts
        
    Returns:
        tuple: (list of original texts, list of generated texts)
    """
    # Load original texts
    originals = []
    for filename in sorted(os.listdir(original_dir)):
        if filename.endswith('.txt'):
            file_path = os.path.join(original_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    originals.append(f.read())
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    # Load generated texts
    try:
        with open(generated_file, 'r', encoding='utf-8') as f:
            generated_data = json.load(f)
        generated_texts = [entry["generated_text"] for entry in generated_data]
    except Exception as e:
        print(f"Error reading generated file {generated_file}: {e}")
        generated_texts = []

    return originals, generated_texts

def extract_stylometric_features(texts):
    """
    Extract stylometric features from a list of texts.
    Features include:
    - Average sentence length
    - Lexical diversity (unique words / total words)
    - Average word length
    - Flesch Reading Ease score
    - Flesch-Kincaid Grade Level
    """
    features = []
    for text in texts:
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        
        avg_sentence_length = np.mean([len(sent.split()) for sent in sentences]) if sentences else 0
        lexical_diversity = len(set(words)) / len(words) if words else 0
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        readability = flesch_reading_ease(text)
        fk_grade = flesch_kincaid_grade(text)

        features.append([avg_sentence_length, lexical_diversity, avg_word_length, readability, fk_grade])
    return np.array(features)

def compare_styles(original_features, generated_features):
    """
    Compare stylometric features using Euclidean distance and cosine similarity.
    """
    scaler = StandardScaler()
    all_features = np.vstack((original_features, generated_features))
    scaled_features = scaler.fit_transform(all_features)

    # Split scaled features
    scaled_originals = scaled_features[:len(original_features)]
    scaled_generated = scaled_features[len(original_features):]

    # Calculate distances
    distances = []
    for orig_feat in scaled_originals:
        row = [euclidean(orig_feat, gen_feat) for gen_feat in scaled_generated]
        distances.append(row)

    # Calculate average distance
    avg_distance = np.mean(distances)
    print(f"Average Stylometric Distance: {avg_distance:.2f}")

    return distances, avg_distance

def visualize_distances(distance_matrix, viz_dir):
    """
    Visualize stylistic distances using a heatmap and save to file.
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(distance_matrix, cmap='coolwarm', interpolation='nearest')
    plt.colorbar(label='Euclidean Distance')
    plt.title("Stylometric Distance between Original and Generated Texts")
    plt.xlabel("Generated Texts")
    plt.ylabel("Original Texts")
    plt.tight_layout()
    plt.savefig(viz_dir / "distance_heatmap.png")
    plt.close()

def cluster_texts(features, n_clusters=2):
    """
    Cluster texts based on their stylistic features.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(features)
    return clusters, kmeans

def visualize_clusters(features, clusters, viz_dir):
    """
    Visualize the clustering of texts based on their first two stylometric features.
    """
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=features[:, 0], y=features[:, 1], hue=clusters)
    plt.title("Clustering of Texts Based on Stylometric Features")
    plt.xlabel("Average Sentence Length")
    plt.ylabel("Lexical Diversity")
    plt.tight_layout()
    plt.savefig(viz_dir / "clustering_analysis.png")
    plt.close()

def analyze_extreme_pairs(distances, originals, generated):
    """
    Identify and display the most similar and most different text pairs.
    """
    # Convert distances to numpy array if it isn't already
    distances = np.array(distances)
    
    min_dist_idx = np.unravel_index(np.argmin(distances), distances.shape)
    max_dist_idx = np.unravel_index(np.argmax(distances), distances.shape)

    print("\nClosest Stylistic Match:")
    print(f"Original: {originals[min_dist_idx[0]][:500]}...")
    print(f"Generated: {generated[min_dist_idx[1]][:500]}...")

    print("\nLeast Similar Pair:")
    print(f"Original: {originals[max_dist_idx[0]][:500]}...")
    print(f"Generated: {generated[max_dist_idx[1]][:500]}...")

    return min_dist_idx, max_dist_idx

# Main experiment flow
def run_stylometric_analysis(original_dir, generated_file, experiment_name="stylometric_analysis"):
    """
    Run stylometric analysis and save results to experiment-specific directory.
    """
    # Create experiment directory
    experiment_dir = Path(RESULTS_DIR) / experiment_name
    experiment_dir.mkdir(exist_ok=True)
    
    # Create subdirectories
    data_dir = experiment_dir / "data"
    viz_dir = experiment_dir / "visualizations"
    for dir_path in [data_dir, viz_dir]:
        dir_path.mkdir(exist_ok=True)

    originals, generated = load_texts(original_dir, generated_file)

    print("Extracting stylometric features for original texts...")
    original_features = extract_stylometric_features(originals)

    print("Extracting stylometric features for generated texts...")
    generated_features = extract_stylometric_features(generated)

    print("Comparing styles...")
    distances, avg_distance = compare_styles(original_features, generated_features)

    # Save distance matrix to CSV
    pd.DataFrame(distances).to_csv(data_dir / "stylometric_distance_matrix.csv", index=False)

    print("Analyzing extreme pairs...")
    min_dist_idx, max_dist_idx = analyze_extreme_pairs(distances, originals, generated)

    # Save detailed results to file
    print("Saving detailed results...")
    with open(data_dir / "stylometric_analysis_report.txt", "w", encoding='utf-8') as f:
        f.write(f"Average Stylometric Distance: {avg_distance:.2f}\n\n")
        f.write("Closest Matches:\n")
        f.write(f"Original: {originals[min_dist_idx[0]]}\n")
        f.write(f"Generated: {generated[min_dist_idx[1]]}\n\n")
        f.write("Least Similar Pairs:\n")
        f.write(f"Original: {originals[max_dist_idx[0]]}\n")
        f.write(f"Generated: {generated[max_dist_idx[1]]}\n")

    print("Visualizing distances...")
    visualize_distances(np.array(distances), viz_dir)

    # Perform clustering analysis
    print("Performing clustering analysis...")
    all_features = np.vstack((original_features, generated_features))
    clusters, _ = cluster_texts(all_features)
    visualize_clusters(all_features, clusters, viz_dir)

    print(f"\nExperiment results saved to {experiment_dir}")
    return distances, avg_distance

if __name__ == "__main__":
    run_stylometric_analysis(
        original_dir="datasets/preprocessed_texts",
        generated_file="datasets/generated_texts/all_outputs.json",
        experiment_name="stylometric_analysis"
    )
