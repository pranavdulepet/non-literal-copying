import os
import re
from pathlib import Path
from typing import List, Optional

def preprocess_gutenberg_texts(input_dir: str, output_dir: str, min_chapter_length: int = 1000) -> None:
    """
    Preprocess texts from Project Gutenberg to:
    - Remove headers/footers
    - Remove content before first chapter
    - Clean up formatting
    - Split into meaningful chunks
    - Handle different chapter formats
    
    Args:
        input_dir (str): Directory containing raw Gutenberg text files
        output_dir (str): Directory to save preprocessed files
        min_chapter_length (int): Minimum length for valid chapter text
    """
    os.makedirs(output_dir, exist_ok=True)

    chapter_patterns = [
        r"CHAPTER [IVXLCMivxlcm]+",    # CHAPTER I, CHAPTER II, etc.
        r"CHAPTER \d+",                 # CHAPTER 1, CHAPTER 2, etc.
        r"Chapter [IVXLCMivxlcm]+",     # Chapter I, Chapter II, etc.
        r"Chapter \d+",                 # Chapter 1, Chapter 2, etc.
        r"\bCHAPTER ONE\b",            # CHAPTER ONE
        r"\bChapter One\b",            # Chapter One
        r"^\d+\.",                     # 1., 2., etc. at start of line
        r"^I\.",                       # I., II., etc. at start of line
        r"^BOOK [IVXLCMivxlcm]+",      # BOOK I, BOOK II, etc.
        r"^Book [IVXLCMivxlcm]+",      # Book I, Book II, etc.
        r"^\* \* \*$",                 # Scene breaks
        r"^PART [IVXLCMivxlcm]+",      # PART I, PART II, etc.
    ]
    
    chapter_pattern = '|'.join(f"({pattern})" for pattern in chapter_patterns)

    # Common metadata patterns to remove
    metadata_patterns = [
        r"\*\*\* START OF.*?\*\*\*",
        r"\*\*\* END OF.*?\*\*\*",
        r"Produced by .*?\n",
        r"\[Illustration:.*?\]",
        r"\[Footnote:.*?\]",
        r"\[Price:.*?\]",
        r"Transcriber's Note:.*?\n",
        r"Project Gutenberg.*?Foundation",
    ]

    for file in os.listdir(input_dir):
        if not file.endswith('.txt'):
            continue
            
        input_path = os.path.join(input_dir, file)
        output_path = os.path.join(output_dir, file)

        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                text = f.read()

            # Remove metadata and formatting
            for pattern in metadata_patterns:
                text = re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE)

            # Normalize whitespace
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'\n\s*\n', '\n\n', text)
            
            # Split into chapters/sections
            chapters = re.split(chapter_pattern, text)
            
            # Filter out short sections and clean each chapter
            cleaned_chapters = []
            for chapter in chapters:
                if chapter and len(chapter.strip()) > min_chapter_length:
                    cleaned = chapter.strip()
                    cleaned = re.sub(r'\s+', ' ', cleaned)
                    cleaned_chapters.append(cleaned)

            # Rejoin cleaned text
            cleaned_text = '\n\n'.join(cleaned_chapters)
            
            # Final cleanup
            cleaned_text = cleaned_text.strip()
            cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)

            # Write cleaned text
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)

            print(f"Preprocessed {file} -> {output_path}")
            
        except Exception as e:
            print(f"Error processing {file}: {e}")

if __name__ == "__main__":
    preprocess_gutenberg_texts("datasets/original_texts", "datasets/preprocessed_texts")