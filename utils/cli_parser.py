import argparse

def get_model_choice():
    parser = argparse.ArgumentParser(description='Select AI model for text generation')
    parser.add_argument('--model', 
                       choices=['openai', 'gemini'], 
                       default='openai',
                       help='Choose AI model (openai or gemini)')
    args = parser.parse_args()
    return args.model 