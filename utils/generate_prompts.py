import os
import csv

def generate_prompts(input_dir, output_file):
    """
    Generate prompts for preprocessed Gutenberg texts.
    """
    prompts = []
    for idx, file in enumerate(os.listdir(input_dir)):
        with open(os.path.join(input_dir, file), 'r', encoding='utf-8') as f:
            text = f.read().strip()
            excerpt = text[:500] 
            prompts.append({
                "id": idx + 1,
                "prompt_type": "summary",
                "prompt_text": f"Summarize this text in 200 words: '{excerpt}'"
            })
            prompts.append({
                "id": idx + 100,
                "prompt_type": "style_transfer",
                "prompt_text": f"Rewrite this text in the style of a science fiction story: '{excerpt}'"
            })
            prompts.append({
                "id": idx + 200,
                "prompt_type": "open_ended",
                "prompt_text": f"Write a continuation for this text: '{excerpt}'"
            })

    # Save prompts to CSV
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ["id", "prompt_type", "prompt_text"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(prompts)

# Example usage
generate_prompts("datasets/preprocessed_texts", "datasets/prompts/all_prompts.csv")
