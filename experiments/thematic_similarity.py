import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from bertopic import BERTopic
import os
import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from scipy.stats import ttest_ind
from pathlib import Path
from utils.config import RESULTS_DIR

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

def extract_themes(texts, n_topics=10):
    """
    Use BERTopic to extract themes from texts.
    
    Args:
        texts: List of text documents
        n_topics: Maximum number of topics to extract (will be adjusted based on corpus size)
    """
    # Adjust n_topics based on corpus size
    n_docs = len(texts)
    n_topics = min(n_topics, max(2, n_docs - 1))  # Ensure at least 2 topics but less than n_docs
    
    try:
        # Configure BERTopic with correct parameters
        model = BERTopic(
            nr_topics=n_topics,
            min_topic_size=1,  # Allow small topics due to potentially small dataset
            verbose=True
        )
        topics, probs = model.fit_transform(texts)
        return model, topics, probs
    except Exception as e:
        print(f"Warning: Error in theme extraction: {str(e)}")
        # Return dummy results if extraction fails
        return None, [0] * len(texts), [[1.0] * n_topics] * len(texts)

def compare_themes(originals, generated):
    """
    Compare themes between original and AI-generated texts using cosine similarity.
    """
    # Combine texts for unified vectorization
    combined_texts = originals + generated
    vectorizer = CountVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform(combined_texts)

    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(vectors[:len(originals)], vectors[len(originals):])
    return similarity_matrix

def visualize_similarity(similarity_matrix, viz_dir):
    """
    Plot a heatmap of thematic similarity and save to file.
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(similarity_matrix, cmap='coolwarm', interpolation='nearest')
    plt.colorbar(label='Cosine Similarity')
    plt.title("Thematic Similarity between Original and Generated Texts")
    plt.xlabel("Generated Texts")
    plt.ylabel("Original Texts")
    plt.savefig(viz_dir / "thematic_similarity_heatmap.png")
    plt.close()

# Main experiment flow
def run_thematic_similarity_experiment(original_dir, generated_file, experiment_name="thematic_similarity"):
    # Create experiment directory
    experiment_dir = Path(RESULTS_DIR) / experiment_name
    experiment_dir.mkdir(exist_ok=True)
    
    # Create subdirectories
    data_dir = experiment_dir / "data"
    viz_dir = experiment_dir / "visualizations"
    for dir_path in [data_dir, viz_dir]:
        dir_path.mkdir(exist_ok=True)

    originals, generated = load_texts(original_dir, generated_file)

    # Extract themes with error handling
    print("Extracting themes for original texts...")
    original_model, original_topics, _ = extract_themes(originals)
    
    if original_model:
        # Save topic visualization only if model creation succeeded
        print("\nSaving topic visualization...")
        try:
            topic_viz = original_model.visualize_topics()
            topic_viz.write_html(str(viz_dir / "original_topics.html"))
            
            print("\nOriginal Topic Labels:")
            original_labels = original_model.get_topic_info()
            original_labels.to_csv(data_dir / "original_topic_labels.csv")
        except Exception as e:
            print(f"Warning: Could not generate topic visualizations: {str(e)}")
    
    # Compare themes using simpler cosine similarity approach
    print("Comparing themes...")
    similarity_matrix = compare_themes(originals, generated)

    # Save similarity matrix
    np.save(data_dir / "similarity_matrix.npy", similarity_matrix)
    
    # Generate report
    print("\nGenerating report...")
    with open(data_dir / "thematic_similarity_report.txt", "w") as report:
        report.write(f"Average Thematic Similarity: {np.mean(similarity_matrix):.2f}\n")
        report.write(f"Number of original texts: {len(originals)}\n")
        report.write(f"Number of generated texts: {len(generated)}\n")

    # Visualize similarity
    visualize_similarity(similarity_matrix, viz_dir)
    
    print(f"\nExperiment results saved to {experiment_dir}")
    return similarity_matrix

# Example usage
if __name__ == "__main__":
    run_thematic_similarity_experiment(
        original_dir="datasets/preprocessed_texts",
        generated_file="datasets/generated_texts/all_outputs.json"
    )
