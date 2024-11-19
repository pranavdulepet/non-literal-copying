import os
import requests
import re

def clean_gutenberg_text(text):
    """Clean Project Gutenberg text by removing headers and footers."""
    # Remove header
    start_markers = [
        "*** START OF THIS PROJECT GUTENBERG",
        "***START OF THIS PROJECT GUTENBERG",
        "*** START OF THE PROJECT GUTENBERG",
        "***START OF THE PROJECT GUTENBERG"
    ]
    
    end_markers = [
        "*** END OF THIS PROJECT GUTENBERG",
        "***END OF THIS PROJECT GUTENBERG",
        "*** END OF THE PROJECT GUTENBERG",
        "***END OF THE PROJECT GUTENBERG"
    ]
    
    text = text.replace('\r\n', '\n')  # Normalize line endings
    
    # Find start of content
    start_pos = 0
    for marker in start_markers:
        pos = text.find(marker)
        if pos != -1:
            start_pos = text.find('\n', pos) + 1
            break
            
    # Find end of content
    end_pos = len(text)
    for marker in end_markers:
        pos = text.find(marker)
        if pos != -1:
            end_pos = pos
            break
            
    return text[start_pos:end_pos].strip()

def fetch_gutenberg_texts(output_dir, num_books=5):
    """
    Fetch texts from Project Gutenberg using direct HTTP requests.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # List of popular book IDs from Project Gutenberg
    book_ids = [1342, 11, 1661, 1952, 84][:num_books]  # Pride & Prejudice, Alice in Wonderland, etc.
    
    for book_id in book_ids:
        try:
            # Try primary URL format
            url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
            response = requests.get(url)
            
            if response.status_code != 200:
                # Try alternative URL format
                url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
                response = requests.get(url)
            
            if response.status_code == 200:
                # Clean and save the text
                text = response.text
                clean_text = clean_gutenberg_text(text)
                
                file_path = os.path.join(output_dir, f"book_{book_id}.txt")
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(clean_text)
                print(f"Downloaded Book ID {book_id} -> {file_path}")
            else:
                print(f"Failed to download Book ID {book_id}: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"Error downloading Book ID {book_id}: {e}")

if __name__ == "__main__":
    fetch_gutenberg_texts("datasets/original_texts", num_books=5)
