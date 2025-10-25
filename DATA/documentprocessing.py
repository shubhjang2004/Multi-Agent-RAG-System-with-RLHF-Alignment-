#  Document Processing



import json
import re
from pathlib import Path
from typing import List, Dict
import fitz  # PyMuPDF
from tqdm import tqdm
import nltk
import spacy
from collections import defaultdict

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

nlp = spacy.load('en_core_web_sm')

PDF_DIR = Path("data/raw/arxiv_papers")
OUTPUT_DIR = Path("data/processed/chunks")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Document processing setup complete")


CONFIG = {
    'chunk_size': 512,
    'chunk_overlap': 128,
    'min_chunk_length': 100,
    'max_chunk_length': 1024,
    'remove_references': True,
    'remove_headers_footers': True,
    'min_words': 20,
}

print("Configuration:")
for k, v in CONFIG.items():
    print(f"  {k}: {v}")


def extract_text_from_pdf(pdf_path):
    """Extract text from PDF using PyMuPDF"""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page_num in range(len(doc)):
            page = doc[page_num]
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        print(f"Error processing {pdf_path.name}: {e}")
        return None

def clean_text(text):
    """Clean extracted text"""
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    # Remove page numbers
    text = re.sub(r'\n\d+\n', '\n', text)
    # Remove excessive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove URLs
    text = re.sub(r'http[s]?://\S+', '', text)
    return text.strip()

def remove_references_section(text):
    """Remove references section from paper"""
    patterns = [
        r'\n\s*References\s*\n',
        r'\n\s*REFERENCES\s*\n',
        r'\n\s*Bibliography\s*\n',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return text[:match.start()]
    return text

def is_good_chunk(chunk, min_words=20):
    """Filter out low-quality chunks"""
    words = chunk.split()
    if len(words) < min_words:
        return False
    # Check for too many numbers (likely tables)
    num_count = sum(1 for w in words if w.replace('.', '').replace(',', '').isdigit())
    if num_count / len(words) > 0.5:
        return False
    # Check for too many special characters
    special_chars = sum(1 for c in chunk if not c.isalnum() and not c.isspace())
    if special_chars / len(chunk) > 0.3:
        return False
    return True


def semantic_chunking(text, chunk_size=512, overlap=128):
    """Chunk text based on sentence boundaries"""
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence.split())
        
        if current_length + sentence_length > chunk_size and current_chunk:
            # Save current chunk
            chunks.append(' '.join(current_chunk))
            
            # Start new chunk with overlap
            overlap_sentences = []
            overlap_length = 0
            for s in reversed(current_chunk):
                s_len = len(s.split())
                if overlap_length + s_len <= overlap:
                    overlap_sentences.insert(0, s)
                    overlap_length += s_len
                else:
                    break
            
            current_chunk = overlap_sentences
            current_length = overlap_length
        
        current_chunk.append(sentence)
        current_length += sentence_length
    
    # Add remaining chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks


def process_document(pdf_path, paper_metadata, config):
    """Process a single document"""
    # Extract text
    text = extract_text_from_pdf(pdf_path)
    if not text:
        return []
    
    # Clean text
    text = clean_text(text)
    
    # Remove references if configured
    if config['remove_references']:
        text = remove_references_section(text)
    
    # Create chunks
    chunks = semantic_chunking(
        text,
        chunk_size=config['chunk_size'],
        overlap=config['chunk_overlap']
    )
    
    # Filter and create chunk metadata
    processed_chunks = []
    for i, chunk in enumerate(chunks):
        if not is_good_chunk(chunk, min_words=config['min_words']):
            continue
        
        chunk_data = {
            'chunk_id': f"{paper_metadata['arxiv_id']}_{i}",
            'text': chunk,
            'arxiv_id': paper_metadata['arxiv_id'],
            'title': paper_metadata['title'],
            'authors': paper_metadata['authors'],
            'abstract': paper_metadata['abstract'],
            'categories': paper_metadata['categories'],
            'published': paper_metadata['published'],
            'chunk_index': i,
            'word_count': len(chunk.split()),
        }
        processed_chunks.append(chunk_data)
    
    return processed_chunks


# Load metadata
with open('data/raw/arxiv_metadata.json', 'r') as f:
    papers_metadata = json.load(f)

# Create metadata lookup
metadata_lookup = {p['arxiv_id']: p for p in papers_metadata}

print(f"Loaded metadata for {len(papers_metadata)} papers")


# Process all PDFs
all_chunks = []
processing_stats = {
    'total_pdfs': 0,
    'successful': 0,
    'failed': 0,
    'total_chunks': 0,
}

pdf_files = list(PDF_DIR.glob("*.pdf"))
print(f"Found {len(pdf_files)} PDF files")

for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
    processing_stats['total_pdfs'] += 1
    
    # Get arxiv_id from filename
    arxiv_id = pdf_path.stem.replace('_', '/')
    
    # Get metadata
    if arxiv_id not in metadata_lookup:
        # Try alternative format
        arxiv_id = pdf_path.stem
        if arxiv_id not in metadata_lookup:
            processing_stats['failed'] += 1
            continue
    
    paper_metadata = metadata_lookup[arxiv_id]
    
    # Process document
    chunks = process_document(pdf_path, paper_metadata, CONFIG)
    
    if chunks:
        all_chunks.extend(chunks)
        processing_stats['successful'] += 1
        processing_stats['total_chunks'] += len(chunks)
    else:
        processing_stats['failed'] += 1

print(f"\n{'='*60}")
print(f"Processing complete!")
print(f"Total PDFs: {processing_stats['total_pdfs']}")
print(f"Successful: {processing_stats['successful']}")
print(f"Failed: {processing_stats['failed']}")
print(f"Total chunks: {processing_stats['total_chunks']}")
print(f"Avg chunks per paper: {processing_stats['total_chunks']/processing_stats['successful']:.1f}")
print(f"{'='*60}")


# Save chunks
chunks_file = OUTPUT_DIR / "all_chunks.json"
with open(chunks_file, 'w') as f:
    json.dump(all_chunks, f, indent=2)

stats_file = OUTPUT_DIR / "processing_stats.json"
with open(stats_file, 'w') as f:
    json.dump(processing_stats, f, indent=2)

print(f"\nSaved {len(all_chunks)} chunks to {chunks_file}")
print(f"Saved stats to {stats_file}")


# Analyze chunks
word_counts = [c['word_count'] for c in all_chunks]
print(f"\nChunk Statistics:")
print(f"  Min words: {min(word_counts)}")
print(f"  Max words: {max(word_counts)}")
print(f"  Mean words: {sum(word_counts)/len(word_counts):.1f}")
print(f"  Total words: {sum(word_counts):,}")

