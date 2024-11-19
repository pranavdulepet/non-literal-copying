import os
import re

def preprocess_gutenberg_texts(input_dir: str, output_dir: str) -> None:
    """
    Preprocess texts from Project Gutenberg to remove headers/footers and content before first chapter.
    
    Args:
        input_dir (str): Directory containing raw Gutenberg text files
        output_dir (str): Directory to save preprocessed files
    """
    os.makedirs(output_dir, exist_ok=True)

    # Common chapter start patterns
    chapter_patterns = [
        r"CHAPTER [IVXLCMivxlcm]+",  # CHAPTER I, CHAPTER II, etc.
        r"CHAPTER \d+",              # CHAPTER 1, CHAPTER 2, etc.
        r"Chapter [IVXLCMivxlcm]+",  # Chapter I, Chapter II, etc.
        r"Chapter \d+",              # Chapter 1, Chapter 2, etc.
        r"\bCHAPTER ONE\b",         # CHAPTER ONE
        r"\bChapter One\b",         # Chapter One
        r"^\d+\.",                  # 1., 2., etc. at start of line
        r"^I\.",                    # I., II., etc. at start of line
    ]
    
    chapter_pattern = '|'.join(chapter_patterns)

    for file in os.listdir(input_dir):
        input_path = os.path.join(input_dir, file)
        output_path = os.path.join(output_dir, file)

        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                text = f.read()

            # Remove Project Gutenberg header and footer
            text = re.sub(r"\*\*\* START OF.*?\*\*\*", "", text, flags=re.DOTALL)
            text = re.sub(r"\*\*\* END OF.*?\*\*\*", "", text, flags=re.DOTALL)

            # Remove non-content sections
            text = re.sub(r"Produced by .*?\n", "", text)
            text = text.strip()

            # Remove content before first chapter
            text = re.sub(chapter_pattern, "", text)

            # Write cleaned text
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text)

            print(f"Preprocessed {file} -> {output_path}")
        except Exception as e:
            print(f"Error processing {file}: {e}")

if __name__ == "__main__":
    preprocess_gutenberg_texts("datasets/original_texts", "datasets/preprocessed_texts")
