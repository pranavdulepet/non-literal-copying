# LLM Non-Literal Copying Analysis

A research framework for analyzing and quantifying non-literal copying in large language model outputs, focusing on thematic, stylistic, and lexical transformations.

## Overview

This project explores the boundaries of non-literal copying in AI-generated text, drawing parallels from legal precedents in AI-generated images. Using a combination of computational analysis and empirical evaluation, this project investigates how LLMs balance creativity with fidelity to source material.

## Key Features

- Thematic similarity analysis between source and generated texts
- Stylometric analysis for detecting writing style patterns
- Creative paraphrasing evaluation
- Trigger analysis for understanding prompt variations
- Automated text preprocessing pipeline for Project Gutenberg texts

## Installation

1. Clone the repository
2. Ensure Python 3.10+
3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── datasets/
│   ├── original_texts/      # Raw Gutenberg texts
│   ├── preprocessed_texts/  # Cleaned text files
│   ├── prompts/            # Generated prompts
│   └── generated_texts/    # LLM outputs
├── experiments/
│   ├── stylometric_analysis.py
│   ├── thematic_similarity.py
│   ├── creative_paraphrasing.py
│   └── trigger_analysis.py
├── utils/
│   ├── fetch_gutenberg.py
│   ├── preprocess_gutenberg.py
│   ├── generate_prompts.py
│   └── generate_outputs.py
└── results/
    └── experiment_results/
```

## Usage

### 1. Data Preparation

First, preprocess the Gutenberg texts:

```python
from utils.preprocess_gutenberg import preprocess_gutenberg_texts

preprocess_gutenberg_texts("datasets/original_texts", "datasets/preprocessed_texts")
```

### 2. Generate Prompts

Generate prompts from preprocessed texts:

```python
from utils.generate_prompts import generate_prompts

generate_prompts("datasets/preprocessed_texts", "datasets/prompts/all_prompts.csv")
```

### 3. Generate LLM Outputs

Generate text using either OpenAI or Google's Gemini models:

```python
from utils.generate_outputs import generate_outputs

generate_outputs("datasets/prompts/all_prompts.csv", "datasets/generated_texts/all_outputs.json")
```

### 4. Run Experiments

Run various analysis experiments:

```python
from experiments.stylometric_analysis import run_stylometric_analysis
from experiments.thematic_similarity import run_thematic_similarity_experiment

# Run stylometric analysis
run_stylometric_analysis(
    original_dir="datasets/preprocessed_texts",
    generated_file="datasets/generated_texts/all_outputs.json",
    experiment_name="stylometric_analysis"
)

# Run thematic similarity analysis
run_thematic_similarity_experiment(
    original_dir="datasets/preprocessed_texts",
    generated_file="datasets/generated_texts/all_outputs.json"
)
```

## Dependencies

Key dependencies include:
- transformers >= 4.11.0
- torch >= 1.9.0
- bertopic >= 0.9.4
- scikit-learn >= 0.24.2
- OpenAI API access
- Google Generative AI API access

## License

This project is licensed under the MIT License - see the LICENSE file for details.
