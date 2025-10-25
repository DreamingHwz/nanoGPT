import os
import re
import unicodedata
from typing import Optional

import pandas as pd

class PoemCSVCleaner:
    def __init__(self, input_csv, output_txt):
        self.input_csv = input_csv
        self.output_txt = output_txt
    
    @staticmethod
    def _strip_edge_blank_lines(lines):
        """Remove leading and trailing empty strings from a list of lines."""

        start = 0
        end = len(lines)

        while start < end and lines[start] == "":
            start += 1

        while end > start and lines[end - 1] == "":
            end -= 1

        return lines[start:end]

    def clean_poem_text(self, poem_text: str) -> str:
        """Heuristically clean a single poem for language modeling."""

        if not isinstance(poem_text, str):
            poem_text = "" if poem_text is None else str(poem_text)

        # Normalise common newline variants and unicode presentation.
        poem_text = poem_text.replace('\r\n', '\n').replace('\r', '\n')
        poem_text = unicodedata.normalize("NFKC", poem_text)

        # Drop editorial placeholders (e.g. "/* Lines 3-7 omitted */").
        poem_text = re.sub(r"/\*.*?\*/", "", poem_text, flags=re.DOTALL)

        # Remove obvious CSV artefacts that occasionally sneak through.
        poem_text = re.sub(r'^"+\s*', '', poem_text)
        poem_text = re.sub(r'\s*"+$', '', poem_text)

        # Trim whitespace at both ends once heavy lifting is done.
        poem_text = poem_text.strip()

        # Remove empty or comment-only lines that remain after placeholder removal.
        cleaned_lines = []
        for raw_line in poem_text.split('\n'):
            line = raw_line.rstrip()
            
            sentinel = line.lstrip()
            if sentinel.startswith('//'):
                continue

            cleaned_lines.append(line)

        poem_text = '\n'.join(cleaned_lines)

        # Collapse runs of blank lines: 1个空行删掉，2个或更多空行改成1个空行
        poem_text = re.sub(r'\n\n', '\n', poem_text)
        poem_text = re.sub(r'\n\n\n+', '\n\n', poem_text)

        # Remove lingering control characters while keeping tabs/newlines.
        poem_text = ''.join(
            ch for ch in poem_text
            if (ch in '\n\t') or unicodedata.category(ch)[0] != 'C'
        )

        # Deduplicate accidental leading/trailing whitespace per line once more.
        poem_text = '\n'.join(part.rstrip() for part in poem_text.split('\n'))

        return poem_text

    def clean_title(self, title_text: Optional[str]) -> str:
        """Normalise poem titles while preserving human-friendly casing."""

        if title_text is None or (isinstance(title_text, float) and pd.isna(title_text)):
            return "Untitled"

        title_text = unicodedata.normalize("NFKC", str(title_text))
        title_text = title_text.strip()
        title_text = re.sub(r'\s+', ' ', title_text)

        if not title_text:
            return "Untitled"

        return title_text

    def format_poem_entry(self, title: str, poem: str) -> str:
        """Return a standardised textual representation ready for training."""

        formatted_poem = poem.strip()
        if not formatted_poem:
            return ""

        return f"title: {title}\npoem: {formatted_poem}\n\n\n"
    
    def clean(self):
        print(f"Reading CSV file: {self.input_csv}")
        
        # Read CSV
        try:
            df = pd.read_csv(self.input_csv)
        except Exception as e:
            print(f"Error reading CSV: {e}")
            return False
        
        print(f"Found {len(df)} poems")
        print(f"Columns: {df.columns.tolist()}")
        
        # Check which columns we have
        if 'Poem' in df.columns:
            poem_column = 'Poem'
        elif 'poem' in df.columns:
            poem_column = 'poem'
        else:
            print(f"Error: 'Poem' column not found")
            return False
        
        # Check for title column
        if 'Title' in df.columns:
            title_column = 'Title'
        elif 'title' in df.columns:
            title_column = 'title'
        else:
            title_column = None
            print("Warning: 'Title' column not found. Poems will be saved without titles.")
        
        print(f"Using poem column: '{poem_column}'")
        if title_column:
            print(f"Using title column: '{title_column}'")
        
        # Extract and combine all poems
        poem_entries = []
        skipped = 0
        empty_after_clean = 0
        
        for idx, row in df.iterrows():
            poem = row[poem_column]
            
            # Skip empty or NaN poems
            if pd.isna(poem) or str(poem).strip() == '':
                skipped += 1
                continue
            
            # Convert to string if needed
            poem = str(poem)
            
            # Clean the poem
            poem = self.clean_poem_text(poem)

            # Drop poems that become empty after cleaning
            if not poem.strip():
                empty_after_clean += 1
                continue
            
            # Add title if available
            if title_column:
                title = self.clean_title(row[title_column])
            else:
                title = "Untitled"

            formatted = self.format_poem_entry(title, poem)
            if formatted:
                poem_entries.append(formatted)
            else:
                empty_after_clean += 1
        
        processed = len(poem_entries)
        print(
            f"Processed {processed} poems "
            f"(skipped {skipped} empty source rows, {empty_after_clean} empty after cleaning)"
        )
        
        # Write to file
        with open(self.output_txt, 'w', encoding='utf-8') as f:
            f.writelines(poem_entries)
        all_poems = ''.join(poem_entries)
        
        # Print statistics
        chars = len(all_poems)
        lines = len([l for l in all_poems.split('\n') if l.strip()])
        
        print(f"\n✓ Cleaning complete!")
        print(f"  Total characters: {chars:,}")
        print(f"  Non-empty lines: {lines:,}")
        print(f"  Saved to: {self.output_txt}")
        
        # Show sample
        print(f"\n--- Sample of output ---")
        sample_end = min(800, len(all_poems))
        print(all_poems[:sample_end])
        print("...")
        
        return True

if __name__ == "__main__":
    # Update these filenames as needed
    input_file = "PoetryFoundationData.csv"  # Your CSV file
    output_file = "cleaned_poems.txt"
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found")
        print("Please place your CSV file in the current directory")
        exit(1)
    
    cleaner = PoemCSVCleaner(input_file, output_file)
    cleaner.clean()