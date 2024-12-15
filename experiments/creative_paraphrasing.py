import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import nltk
import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bert_score import score as bert_score
import matplotlib.pyplot as plt
from utils.api_interface import get_openai_response, get_gemini_response, get_dual_responses
from scipy.stats import ttest_rel
from utils.config import PREPROCESSED_TEXTS_DIR, RESULTS_DIR
import os

nltk.download('punkt')

def generate_paraphrases(original_texts, model="gpt-4", temperature=0.7, use_both_models=False):
    """
    Generate paraphrases using OpenAI GPT-4 and/or Google Gemini.
    """
    paraphrases = []
    total = len(original_texts)
    for idx, text in enumerate(original_texts):
        print(f"Generating paraphrase for text {idx+1}/{total}...")
        try:
            if use_both_models:
                responses = get_openai_response(f"Paraphrase this text: '{text}'") #get_dual_responses(f"Paraphrase this text: '{text}'")
                paraphrases.append(responses)
            else:
                paraphrase = get_openai_response(
                    f"Paraphrase this text: '{text}'",
                    model=model,
                    temperature=temperature
                )
                paraphrases.append(paraphrase)
        except Exception as e:
            print(f"Error generating paraphrase for text {idx+1}: {e}")
            paraphrases.append("")
    return paraphrases

def calculate_lexical_similarity(original, paraphrased):
    """
    Calculate lexical similarity using BLEU and cosine similarity.
    """
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform([original, paraphrased])
    cosine_sim = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return cosine_sim

def calculate_semantic_similarity(original, paraphrased):
    """
    Calculate semantic similarity using BERTScore.
    """
    P, R, F1 = bert_score([paraphrased], [original], lang="en", verbose=False)
    return F1.mean().item()

def calculate_novelty(original, paraphrased):
    """
    Calculate novelty as the proportion of new words in the paraphrased text.
    Returns a value between 0 (no new words) and 1 (all new words).
    """
    original_words = set(original.lower().split())
    paraphrased_words = set(paraphrased.lower().split())
    return len(paraphrased_words - original_words) / len(paraphrased_words)

def evaluate_creativity(original_texts, paraphrased_texts):
    """
    Evaluate paraphrases using lexical similarity, semantic similarity, and novelty.
    """
    lexical_similarities = []
    semantic_similarities = []
    novelty_scores = []

    for original, paraphrased in zip(original_texts, paraphrased_texts):
        lexical_sim = calculate_lexical_similarity(original, paraphrased)
        semantic_sim = calculate_semantic_similarity(original, paraphrased)
        novelty = calculate_novelty(original, paraphrased)
        
        lexical_similarities.append(lexical_sim)
        semantic_similarities.append(semantic_sim)
        novelty_scores.append(novelty)

    return lexical_similarities, semantic_similarities, novelty_scores

def visualize_results(lexical_similarities, semantic_similarities, novelty_scores, viz_dir):
    """
    Visualize the relationships between lexical similarity, semantic similarity, and novelty.
    Save all plots to the visualization directory.
    """
    # Create a figure with 2 rows and 2 columns
    fig = plt.figure(figsize=(15, 12))
    
    # Plot 1: Distributions (Histogram)
    ax1 = plt.subplot(2, 2, 1)
    ax1.hist(lexical_similarities, bins=10, alpha=0.7, label='Lexical Similarity')
    ax1.hist(semantic_similarities, bins=10, alpha=0.7, label='Semantic Similarity')
    ax1.hist(novelty_scores, bins=10, alpha=0.7, label='Novelty')
    ax1.legend(loc='upper right')
    ax1.set_title("Distribution of Metrics")
    ax1.set_xlabel("Score")
    ax1.set_ylabel("Frequency")
    
    # Plot 2: Lexical vs Semantic (with novelty as color)
    ax2 = plt.subplot(2, 2, 2)
    scatter = ax2.scatter(lexical_similarities, semantic_similarities, 
                         c=novelty_scores, cmap='viridis', alpha=0.7)
    ax2.set_title("Lexical vs. Semantic Similarity\n(color indicates novelty)")
    ax2.set_xlabel("Lexical Similarity")
    ax2.set_ylabel("Semantic Similarity")
    plt.colorbar(scatter, ax=ax2, label='Novelty Score')
    ax2.grid(True)
    
    # Plot 3: Metrics over samples
    ax3 = plt.subplot(2, 2, (3, 4))  # Spans both bottom grid positions
    x = range(len(lexical_similarities))
    ax3.plot(x, lexical_similarities, 'b-', label='Lexical Similarity', alpha=0.7)
    ax3.plot(x, semantic_similarities, 'r-', label='Semantic Similarity', alpha=0.7)
    ax3.plot(x, novelty_scores, 'g-', label='Novelty', alpha=0.7)
    ax3.set_title("Metrics Across Samples")
    ax3.set_xlabel("Sample Index")
    ax3.set_ylabel("Score")
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    
    # Save the combined plot
    plt.savefig(viz_dir / "metrics_overview.png")
    plt.close()
    
    # Save individual metric distributions
    for metric, values in [
        ("lexical", lexical_similarities),
        ("semantic", semantic_similarities),
        ("novelty", novelty_scores)
    ]:
        plt.figure(figsize=(8, 6))
        plt.hist(values, bins=15, alpha=0.7)
        plt.title(f"{metric.title()} Similarity Distribution")
        plt.xlabel("Score")
        plt.ylabel("Frequency")
        plt.savefig(viz_dir / f"{metric}_distribution.png")
        plt.close()

def run_statistical_tests(lexical_similarities, semantic_similarities, novelty_scores):
    """
    Run statistical tests to analyze lexical similarity, semantic similarity, and novelty distributions.
    Tests whether each metric significantly differs from its inverse (1 - score).
    """
    # Run t-tests for each metric
    t_stat_lexical, p_val_lexical = ttest_rel(lexical_similarities, [1 - x for x in lexical_similarities])
    t_stat_semantic, p_val_semantic = ttest_rel(semantic_similarities, [1 - x for x in semantic_similarities])
    t_stat_novelty, p_val_novelty = ttest_rel(novelty_scores, [1 - x for x in novelty_scores])

    print("\nStatistical Test Results:")
    print("-" * 50)
    print(f"Lexical Similarity T-test: t={t_stat_lexical:.2f}, p={p_val_lexical:.3f}")
    print(f"Semantic Similarity T-test: t={t_stat_semantic:.2f}, p={p_val_semantic:.3f}")
    print(f"Novelty T-test: t={t_stat_novelty:.2f}, p={p_val_novelty:.3f}")
    
    return {
        "lexical": {"t_stat": t_stat_lexical, "p_value": p_val_lexical},
        "semantic": {"t_stat": t_stat_semantic, "p_value": p_val_semantic},
        "novelty": {"t_stat": t_stat_novelty, "p_value": p_val_novelty}
    }

def identify_creative_examples(original_texts, paraphrased_texts, lexical_similarities, 
                             semantic_similarities, novelty_scores, top_k=3):
    """
    Identify the most creative paraphrases based on high semantic similarity and high novelty.
    """
    creativity_scores = [(sem + nov - lex) / 2 
                        for sem, nov, lex in zip(semantic_similarities, novelty_scores, lexical_similarities)]
    
    # Get indices of top k creative examples
    top_indices = np.argsort(creativity_scores)[-top_k:][::-1]
    
    creative_examples = []
    for idx in top_indices:
        example = {
            "original": original_texts[idx],
            "paraphrased": paraphrased_texts[idx],
            "metrics": {
                "lexical_similarity": lexical_similarities[idx],
                "semantic_similarity": semantic_similarities[idx],
                "novelty": novelty_scores[idx],
                "creativity_score": creativity_scores[idx]
            }
        }
        creative_examples.append(example)
    
    return creative_examples

def save_creative_paraphrasing_report(filename, results, stats):
    """
    Save a comprehensive summary report for the creative paraphrasing experiment.
    """
    with open(filename, 'w') as f:
        f.write("# Creative Paraphrasing Experiment Report\n\n")
        
        # Overall Metrics Summary
        f.write("## Metrics Summary\n")
        f.write(f"Number of samples analyzed: {len(results['metrics']['lexical_similarities'])}\n\n")
        f.write("### Averages\n")
        f.write(f"- Average Lexical Similarity: {np.mean(results['metrics']['lexical_similarities']):.3f}\n")
        f.write(f"- Average Semantic Similarity: {np.mean(results['metrics']['semantic_similarities']):.3f}\n")
        f.write(f"- Average Novelty Score: {np.mean(results['metrics']['novelty_scores']):.3f}\n\n")
        
        # Statistical Analysis
        f.write("## Statistical Analysis\n")
        f.write("### T-Tests (comparing each metric against its inverse)\n")
        for metric, data in stats.items():
            f.write(f"#### {metric.title()} Similarity\n")
            f.write(f"- t-statistic: {data['t_stat']:.3f}\n")
            f.write(f"- p-value: {data['p_value']:.3f}\n")
            f.write(f"- Significance: {'Significant' if data['p_value'] < 0.05 else 'Not significant'} at α=0.05\n\n")
        
        # Creative Examples
        f.write("## Most Creative Examples\n")
        for i, example in enumerate(results['creative_examples'], 1):
            f.write(f"\n### Example {i}\n")
            f.write(f"**Original Text:**\n{example['original']}\n\n")
            f.write(f"**Paraphrased Text:**\n{example['paraphrased']}\n\n")
            f.write("**Metrics:**\n")
            f.write(f"- Lexical Similarity: {example['metrics']['lexical_similarity']:.3f}\n")
            f.write(f"- Semantic Similarity: {example['metrics']['semantic_similarity']:.3f}\n")
            f.write(f"- Novelty Score: {example['metrics']['novelty']:.3f}\n")
            f.write(f"- Creativity Score: {example['metrics']['creativity_score']:.3f}\n\n")

def run_creative_paraphrasing_experiment(original_file, experiment_name="creative_paraphrasing", max_sections=10):
    """
    Run the complete creative paraphrasing experiment and save all outputs to a dedicated folder.
    
    Args:
        original_file: Path to the input file
        experiment_name: Name for the experiment outputs
        max_sections: Maximum number of sections to process (default: 10)
    """
    # Create experiment directory
    experiment_dir = Path(RESULTS_DIR) / experiment_name
    experiment_dir.mkdir(exist_ok=True)
    
    # Create subdirectories
    data_dir = experiment_dir / "data"
    viz_dir = experiment_dir / "visualizations"
    for dir_path in [data_dir, viz_dir]:
        dir_path.mkdir(exist_ok=True)
    
    # Define output paths
    results_file = data_dir / "paraphrasing_results.json"
    report_file = data_dir / "analysis_report.md"
    metrics_file = data_dir / "metrics.json"
    
    # Step 1: Load original texts (with limit)
    with open(original_file, 'r', encoding='utf-8') as f:
        original_texts = [line.strip() for line in f.readlines() if line.strip()][:max_sections]

    # Step 2: Generate paraphrases
    print("\nGenerating paraphrases...")
    paraphrased_texts = generate_paraphrases(original_texts)

    # Step 3: Evaluate creativity
    print("\nEvaluating creativity metrics...")
    lexical_similarities, semantic_similarities, novelty_scores = evaluate_creativity(
        original_texts, paraphrased_texts
    )

    # Step 4: Run statistical tests
    print("\nRunning statistical analysis...")
    statistical_tests = run_statistical_tests(
        lexical_similarities, semantic_similarities, novelty_scores
    )

    # Step 5: Identify creative examples
    print("\nIdentifying most creative examples...")
    creative_examples = identify_creative_examples(
        original_texts, paraphrased_texts,
        lexical_similarities, semantic_similarities, novelty_scores
    )

    # Save metrics
    metrics = {
        "averages": {
            "lexical_similarity": float(np.mean(lexical_similarities)),
            "semantic_similarity": float(np.mean(semantic_similarities)),
            "novelty": float(np.mean(novelty_scores))
        },
        "statistical_tests": statistical_tests
    }
    
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)

    # Save complete results
    results = {
        "original_texts": original_texts,
        "paraphrased_texts": paraphrased_texts,
        "metrics": {
            "lexical_similarities": lexical_similarities,
            "semantic_similarities": semantic_similarities,
            "novelty_scores": novelty_scores
        },
        "statistical_tests": statistical_tests,
        "creative_examples": creative_examples
    }

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)

    # Generate and save report
    save_creative_paraphrasing_report(report_file, results, statistical_tests)

    # Save visualizations
    print("\nGenerating and saving visualizations...")
    visualize_results(
        lexical_similarities, 
        semantic_similarities, 
        novelty_scores,
        viz_dir
    )
    
    print(f"\nExperiment results saved to {experiment_dir}")
    return results

if __name__ == "__main__":
    run_creative_paraphrasing_experiment(
        original_file=PREPROCESSED_TEXTS_DIR / "book_84.txt", 
        experiment_name="creative_paraphrasing"
    )
