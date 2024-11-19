import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from utils.api_interface import get_openai_response, get_gemini_response
from utils.cli_parser import get_model_choice
from utils.config import DEFAULT_TEMPERATURE
import csv
import json

def generate_outputs(prompt_file, output_file, temperature=DEFAULT_TEMPERATURE):
    """
    Generate text outputs from selected AI model.
    """
    model_choice = get_model_choice()
    
    with open(prompt_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        prompts = list(reader)

    outputs = []
    for prompt in prompts:
        # print(f"Generating for Prompt ID {prompt['id']}: {prompt['prompt_text']}")
        try:
            if model_choice == 'openai':
                response_text = get_openai_response(
                    prompt['prompt_text'], 
                    temperature=temperature
                )
            else:
                response_text = get_gemini_response(
                    prompt['prompt_text'],
                    temperature=temperature
                )
                
            outputs.append({
                "prompt_id": prompt["id"],
                "prompt_text": prompt["prompt_text"],
                "generated_text": response_text,
                "model_used": model_choice
            })
        except Exception as e:
            print(f"Error generating for prompt ID {prompt['id']}: {e}")

    # Save outputs to JSON
    with open(output_file, 'w') as jsonfile:
        json.dump(outputs, jsonfile, indent=4)
    print(f"Saved outputs to {output_file}")

# Example usage
generate_outputs("datasets/prompts/all_prompts.csv", "datasets/generated_texts/all_outputs.json")
