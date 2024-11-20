import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import json
from datetime import datetime
import numpy as np

from experiments.creative_paraphrasing import run_creative_paraphrasing_experiment
from experiments.stylometric_analysis import run_stylometric_analysis
from experiments.thematic_similarity import run_thematic_similarity_experiment
from experiments.trigger_analysis import run_trigger_analysis
from utils.config import RESULTS_DIR

# Create base results directory for all experiments
BASE_RESULTS_DIR = RESULTS_DIR / "ran_all"
PREPROCESSED_TEXTS_DIR = Path("datasets/preprocessed_texts")
GENERATED_TEXTS_DIR = Path("datasets/generated_texts")

def convert_numpy_to_native(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_to_native(item) for item in obj]
    return obj

def run_all_experiments(input_text_path=None):
    """Run all experiments and generate unified report."""
    # Create unified results directory
    unified_dir = BASE_RESULTS_DIR / "combined_results"
    unified_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories for each experiment type
    experiment_dirs = {
        'creative_paraphrasing': unified_dir / 'creative_paraphrasing',
        'stylometric_analysis': unified_dir / 'stylometric_analysis',
        'thematic_similarity': unified_dir / 'thematic_similarity',
        'trigger_analysis': unified_dir / 'trigger_analysis'
    }
    
    for dir_path in experiment_dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    results = {}
    
    try:
        # 1. Creative Paraphrasing
        print("\n=== Running Creative Paraphrasing Experiment ===")
        cp_results = run_creative_paraphrasing_experiment(
            original_file=input_text_path or PREPROCESSED_TEXTS_DIR / "book_84.txt",
            experiment_name=str(experiment_dirs['creative_paraphrasing'])
        )
        results['creative_paraphrasing'] = cp_results

        # 2. Stylometric Analysis
        print("\n=== Running Stylometric Analysis ===")
        style_results = run_stylometric_analysis(
            original_dir=PREPROCESSED_TEXTS_DIR,
            generated_file=GENERATED_TEXTS_DIR / "all_outputs.json",
            experiment_name=str(experiment_dirs['stylometric_analysis'])
        )
        results['stylometric_analysis'] = style_results

        # 3. Thematic Similarity
        print("\n=== Running Thematic Similarity Analysis ===")
        thematic_results = run_thematic_similarity_experiment(
            original_dir=PREPROCESSED_TEXTS_DIR,
            generated_file=GENERATED_TEXTS_DIR / "all_outputs.json",
            experiment_name=str(experiment_dirs['thematic_similarity'])
        )
        results['thematic_similarity'] = thematic_results

        # 4. Trigger Analysis
        print("\n=== Running Trigger Analysis ===")
        base_prompt = "Write a continuation of this text using {variation}: 'It was a dark and stormy night.'"
        variations = [
            "a neutral tone",
            "a romantic style",
            "Shakespearean language"
        ]
        trigger_results = run_trigger_analysis(
            base_prompt=base_prompt,
            variations=variations,
            experiment_name=str(experiment_dirs['trigger_analysis'])
        )
        results['trigger_analysis'] = trigger_results

        # Convert numpy arrays to native Python types before saving
        results = convert_numpy_to_native(results)
        
        # Save all results
        results_file = unified_dir / "combined_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nAll experiments completed successfully!")
        print(f"Results saved to: {unified_dir}")
        
        return results

    except Exception as e:
        print(f"\nError running experiments: {str(e)}")
        raise

if __name__ == "__main__":
    run_all_experiments() 