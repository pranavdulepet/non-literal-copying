import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import json
import time
from pathlib import Path
import csv
from openai import OpenAI
from tqdm import tqdm
from utils.config import (
    OPENAI_API_KEY,
    PROMPTS_DIR,
    GENERATED_TEXTS_DIR,
    DEFAULT_TEMPERATURE,
    MAX_TOKENS
)

def generate_outputs(prompts_file, output_file, model="gpt-3.5-turbo", temperature=DEFAULT_TEMPERATURE):
    """
    Generate text outputs using OpenAI's API for each prompt.
    
    Args:
        prompts_file: Path to CSV file containing prompts
        output_file: Path to save JSON outputs
        model: OpenAI model to use (default: gpt-3.5-turbo)
        temperature: Generation temperature (default: from config)
    """
    # Initialize OpenAI client
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    # Load prompts
    prompts = []
    with open(prompts_file, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        prompts = list(reader)
    
    outputs = []
    
    # Generate text for each prompt
    for prompt in tqdm(prompts, desc="Generating outputs"):
        try:
            # Create chat completion
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a creative writing assistant."},
                    {"role": "user", "content": prompt['prompt_text']}
                ],
                temperature=temperature,
                max_tokens=MAX_TOKENS
            )
            
            # Extract generated text
            generated_text = response.choices[0].message.content
            
            outputs.append({
                "prompt_id": prompt['id'],
                "prompt_type": prompt['prompt_type'],
                "prompt_text": prompt['prompt_text'],
                "generated_text": generated_text,
                "model_used": model
            })
            
            # Add small delay to avoid rate limits
            time.sleep(1)
            
        except Exception as e:
            print(f"Error generating for prompt ID {prompt['id']}: {str(e)}")
            continue
    
    # Save outputs
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(outputs, f, indent=4, ensure_ascii=False)
    
    print(f"Saved outputs to {output_file}")

if __name__ == "__main__":
    prompts_file = PROMPTS_DIR / "all_prompts.csv"
    output_file = GENERATED_TEXTS_DIR / "all_outputs.json"
    
    generate_outputs(
        prompts_file=prompts_file,
        output_file=output_file,
        model="gpt-4"
    )
