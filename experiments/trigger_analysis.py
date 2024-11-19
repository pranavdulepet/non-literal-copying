import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import json
from utils.api_interface import get_openai_response, get_gemini_response
from utils.config import MODEL_NAME, DEFAULT_TEMPERATURE, RESULTS_DIR
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import ttest_ind

# --- Data Processing Functions ---

def generate_with_prompt_variations(base_prompt, variations, model=MODEL_NAME, temperature=DEFAULT_TEMPERATURE):
    """Generate AI outputs for a base prompt and its variations."""
    outputs = []
    # Clean up model name by removing comments and whitespace
    model = model.split('#')[0].strip()
    
    for idx, variation in enumerate(variations):
        prompt = base_prompt.format(variation=variation)
        print(f"Generating output for variation {idx+1}: {prompt}")
        try:
            if "gpt" in model.lower():
                # Ensure proper GPT model name format
                if model.lower() == "gpt":
                    model = "gpt-4"
                elif not model.startswith("gpt-"):
                    model = f"gpt-{model}"
                    
                response_text = get_openai_response(
                    prompt, 
                    model=model,
                    temperature=temperature
                )
            else:
                response_text = get_gemini_response(
                    prompt,
                    temperature=temperature
                )
            
            outputs.append({
                "variation": variation,
                "prompt": prompt,
                "generated_text": response_text,
                "model_used": model
            })
        except Exception as e:
            print(f"Error generating for variation {idx+1}: {e}")
            outputs.append({
                "variation": variation,
                "prompt": prompt,
                "generated_text": f"Error: {str(e)}",
                "model_used": model
            })
    return outputs

def save_trigger_analysis_results(outputs, model_name, base_prompt, variations, experiment_dir):
    """Save analysis results to the experiment directory."""
    data_dir = experiment_dir / "data"
    viz_dir = experiment_dir / "visualizations"
    for dir_path in [data_dir, viz_dir]:
        dir_path.mkdir(exist_ok=True)
    
    results = {
        "model_used": model_name,
        "base_prompt": base_prompt,
        "variations": variations,
        "outputs": outputs
    }
    
    with open(data_dir / "trigger_analysis_results.json", 'w') as f:
        json.dump(results, f, indent=4)
        
    return data_dir, viz_dir

# --- Analysis Functions ---

def thematic_similarity(original_text, generated_text):
    """Calculate cosine similarity between original and generated texts."""
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([original_text, generated_text])
    similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return similarity

def analyze_thematic_similarities(base_text, outputs, viz_dir):
    """Analyze and visualize thematic similarities between original and generated texts."""
    similarities = []
    labels = []
    
    for output in outputs:
        similarity = thematic_similarity(base_text, output['generated_text'])
        similarities.append(similarity)
        labels.append(output['variation'])
    
    plt.figure(figsize=(10, 6))
    plt.bar(labels, similarities)
    plt.title("Thematic Similarity to Original Text")
    plt.xlabel("Writing Style")
    plt.ylabel("Cosine Similarity")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(viz_dir / "thematic_similarity_analysis.png")
    plt.close()
    
    return similarities

def statistical_analysis(metric_values, variations):
    """Perform statistical analysis on metric values across variations."""
    results = []
    for i in range(len(variations) - 1):
        for j in range(i + 1, len(variations)):
            try:
                t_stat, p_val = ttest_ind([metric_values[i]], [metric_values[j]])
                # Convert numpy values and handle NaN
                t_stat = float(t_stat) if not np.isnan(t_stat) else None
                p_val = float(p_val) if not np.isnan(p_val) else None
                
                comparison = {
                    "variation1": variations[i],
                    "variation2": variations[j],
                    "t_statistic": t_stat,
                    "p_value": p_val,
                    "significant": bool(p_val < 0.05) if p_val is not None else None
                }
                results.append(comparison)
                print(f"T-test between {variations[i]} and {variations[j]}: "
                      f"t={t_stat if t_stat is not None else 'nan'}, "
                      f"p={p_val if p_val is not None else 'nan'}")
            except Exception as e:
                print(f"Error in statistical comparison: {e}")
                comparison = {
                    "variation1": variations[i],
                    "variation2": variations[j],
                    "t_statistic": None,
                    "p_value": None,
                    "significant": None,
                    "error": str(e)
                }
                results.append(comparison)
    return results

# --- Visualization Functions ---

def visualize_trigger_effect(outputs, viz_dir):
    """Visualize the effect of trigger words/phrases on the output text."""
    lengths = [len(output['generated_text'].split()) for output in outputs]
    trigger_effect = {output['variation']: length for output, length in zip(outputs, lengths)}

    plt.figure(figsize=(10, 6))
    plt.bar(trigger_effect.keys(), trigger_effect.values())
    plt.title("Effect of Triggers on Output Length")
    plt.xlabel("Trigger Variation")
    plt.ylabel("Generated Text Length")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(viz_dir / "trigger_effect_analysis.png")
    plt.close()

def analyze_metrics(outputs, viz_dir):
    """Analyze various metrics across variations."""
    variations = [output['variation'] for output in outputs]
    lengths = [len(output['generated_text'].split()) for output in outputs]
    similarities = [output.get('thematic_similarity', 0) for output in outputs]
    
    metrics = {
        'length': lengths,
        'similarity': similarities
    }
    
    statistical_results = {}
    for metric_name, metric_values in metrics.items():
        print(f"\nStatistical Analysis for {metric_name.title()}:")
        stat_results = statistical_analysis(metric_values, variations)
        statistical_results[metric_name] = stat_results
    
    return statistical_results

# --- Main Function ---

def run_trigger_analysis(base_prompt, variations, model_name=MODEL_NAME, experiment_name="trigger_analysis"):
    """Main experiment flow with organized output saving."""
    experiment_dir = Path(RESULTS_DIR) / experiment_name
    experiment_dir.mkdir(exist_ok=True)
    
    # Step 1: Generate outputs
    outputs = generate_with_prompt_variations(base_prompt, variations, model=model_name)
    
    # Step 2: Save initial results
    data_dir, viz_dir = save_trigger_analysis_results(
        outputs, model_name, base_prompt, variations, experiment_dir
    )
    
    # Step 3: Run analyses (removed SHAP analysis)
    print("Visualizing trigger effects...")
    visualize_trigger_effect(outputs, viz_dir)
    
    print("Analyzing thematic similarities...")
    base_text = base_prompt.split(": '")[1].rstrip("'")
    similarities = analyze_thematic_similarities(base_text, outputs, viz_dir)
    
    # Update outputs with similarity scores
    for output, similarity in zip(outputs, similarities):
        output['thematic_similarity'] = float(similarity)
    
    print("\nPerforming statistical analysis...")
    statistical_results = analyze_metrics(outputs, viz_dir)
    
    # Save final results
    final_results = {
        "model_used": model_name,
        "base_prompt": base_prompt,
        "variations": variations,
        "outputs": outputs,
        "statistical_analysis": statistical_results
    }
    
    with open(data_dir / "trigger_analysis_results.json", 'w') as f:
        json.dump(final_results, f, indent=4)
    
    # Generate summary report with None value handling
    with open(data_dir / "analysis_summary.txt", 'w') as f:
        f.write("Trigger Analysis Summary\n")
        f.write("======================\n\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Base Prompt: {base_prompt}\n\n")
        
        f.write("Statistical Analysis Results:\n")
        for metric, results in statistical_results.items():
            f.write(f"\n{metric.title()} Analysis:\n")
            for result in results:
                f.write(f"- {result['variation1']} vs {result['variation2']}:\n")
                t_stat = result['t_statistic']
                p_val = result['p_value']
                sig = result['significant']
                
                f.write(f"  t-statistic: {t_stat:.2f if t_stat is not None else 'N/A'}\n")
                f.write(f"  p-value: {p_val:.3f if p_val is not None else 'N/A'}\n")
                f.write(f"  significant: {sig if sig is not None else 'N/A'}\n")
    
    print(f"\nExperiment results saved to {experiment_dir}")
    return outputs, statistical_results

if __name__ == "__main__":
    base_prompt = "Write a continuation of this text using {variation}: 'It was a dark and stormy night.'"
    variations = [
        "a neutral tone",
        "a romantic style",
        "Shakespearean language",
        "a futuristic sci-fi setting",
        "imitate the style of J.K. Rowling"
    ]
    run_trigger_analysis(base_prompt, variations, model_name=MODEL_NAME)
