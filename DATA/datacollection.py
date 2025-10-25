#  Data Collection

import arxiv
import json
import os
from pathlib import Path
from tqdm import tqdm
import time
from datetime import datetime, timedelta
import requests

DATA_DIR = Path("data/raw/arxiv_papers")
DATA_DIR.mkdir(parents=True, exist_ok=True)
print(f"Data will be saved to: {DATA_DIR.absolute()}")


CONFIG = {
    'categories': ['cs.LG', 'cs.AI', 'cs.CL', 'cs.CV'],
    'max_papers_per_category': 2500,
    'start_date': '2023-01-01',
    'end_date': '2025-01-01',
    'download_pdfs': True,
    'max_retries': 3,
    'rate_limit_delay': 3
}

print("Configuration:")
for key, value in CONFIG.items():
    print(f"  {key}: {value}")


def search_arxiv_papers(category, max_results=2500, start_date='2023-01-01', end_date='2025-01-01'):
    query = f"cat:{category}"
    client = arxiv.Client(page_size=100, delay_seconds=3, num_retries=3)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    
    papers = []
    print(f"\nSearching category: {category}")
    
    for result in tqdm(client.results(search), total=max_results, desc=f"{category}"):
        published_date = result.published.strftime('%Y-%m-%d')
        if published_date < start_date or published_date > end_date:
            continue
        
        paper_data = {
            'arxiv_id': result.entry_id.split('/')[-1],
            'title': result.title,
            'authors': [author.name for author in result.authors],
            'abstract': result.summary,
            'categories': result.categories,
            'primary_category': result.primary_category,
            'published': published_date,
            'updated': result.updated.strftime('%Y-%m-%d'),
            'pdf_url': result.pdf_url,
            'comment': result.comment if hasattr(result, 'comment') else None,
            'journal_ref': result.journal_ref if hasattr(result, 'journal_ref') else None,
        }
        papers.append(paper_data)
        
        if len(papers) >= max_results:
            break
    
    print(f"Found {len(papers)} papers in {category}")
    return papers

def download_pdf(paper_data, save_dir, max_retries=3):
    arxiv_id = paper_data['arxiv_id']
    pdf_url = paper_data['pdf_url']
    safe_id = arxiv_id.replace('/', '_').replace(':', '_')
    pdf_path = save_dir / f"{safe_id}.pdf"
    
    if pdf_path.exists():
        return True
    
    for attempt in range(max_retries):
        try:
            response = requests.get(pdf_url, timeout=30)
            response.raise_for_status()
            with open(pdf_path, 'wb') as f:
                f.write(response.content)
            return True
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"\nFailed to download {arxiv_id}: {e}")
                return False
            time.sleep(2 ** attempt)
    return False


all_papers = []
for category in CONFIG['categories']:
    papers = search_arxiv_papers(
        category=category,
        max_results=CONFIG['max_papers_per_category'],
        start_date=CONFIG['start_date'],
        end_date=CONFIG['end_date']
    )
    all_papers.extend(papers)
    time.sleep(CONFIG['rate_limit_delay'])

print(f"\n{'='*60}")
print(f"Total papers collected: {len(all_papers)}")
print(f"{'='*60}")


unique_papers = {}
for paper in all_papers:
    arxiv_id = paper['arxiv_id']
    if arxiv_id not in unique_papers:
        unique_papers[arxiv_id] = paper
all_papers = list(unique_papers.values())
print(f"After deduplication: {len(all_papers)} unique papers")

# Cell 7
metadata_path = Path("data/raw/arxiv_metadata.json")
metadata_path.parent.mkdir(parents=True, exist_ok=True)
with open(metadata_path, 'w') as f:
    json.dump(all_papers, f, indent=2)
print(f"Metadata saved to: {metadata_path.absolute()}")

# Cell 8
if CONFIG['download_pdfs']:
    print("\nDownloading PDFs...")
    successful_downloads = 0
    failed_downloads = 0
    
    for paper in tqdm(all_papers, desc="Downloading PDFs"):
        success = download_pdf(paper, DATA_DIR, max_retries=CONFIG['max_retries'])
        if success:
            successful_downloads += 1
        else:
            failed_downloads += 1
        time.sleep(CONFIG['rate_limit_delay'])
    
    print(f"\n{'='*60}")
    print(f"Download complete!")
    print(f"Successful: {successful_downloads}")
    print(f"Failed: {failed_downloads}")
    print(f"{'='*60}")
else:
    print("\nPDF download skipped")


from collections import Counter
primary_categories = [p['primary_category'] for p in all_papers]
category_counts = Counter(primary_categories)

print("\nPapers by Category:")
for cat, count in category_counts.most_common():
    print(f"  {cat}: {count}")

publication_years = [p['published'][:4] for p in all_papers]
year_counts = Counter(publication_years)

print("\nPapers by Year:")
for year in sorted(year_counts.keys()):
    print(f"  {year}: {year_counts[year]}")

print(f"\nTotal unique papers: {len(all_papers)}")
