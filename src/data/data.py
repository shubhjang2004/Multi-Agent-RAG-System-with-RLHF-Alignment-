import argparse
import json
import time
from pathlib import Path

import arxiv
import requests
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Collect ArXiv papers")
    parser.add_argument("--categories",    nargs="+", default=["cs.LG", "cs.AI", "cs.CL", "cs.CV"])
    parser.add_argument("--max_papers",    type=int,  default=2500)
    parser.add_argument("--start_date", type=str, default="2020-01-01")
    parser.add_argument("--end_date",   type=str, default="2026-01-01")
    parser.add_argument("--output_dir",    type=str,  default="data/raw/arxiv_papers")
    parser.add_argument("--metadata_path", type=str,  default="data/raw/arxiv_metadata.json")
    parser.add_argument("--no_pdf",        action="store_true")
    parser.add_argument("--rate_limit",    type=float, default=3.0)
    parser.add_argument("--max_retries",   type=int,  default=3)
    return parser.parse_args()


def search_category(category, max_papers, start_date, end_date, rate_limit):
    client = arxiv.Client(page_size=100, delay_seconds=rate_limit, num_retries=3)
    search = arxiv.Search(
        query=f"cat:{category}",
        max_results=max_papers,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    papers = []
    for result in tqdm(client.results(search), total=max_papers, desc=category):
        published = result.published.strftime("%Y-%m-%d")
        
        papers.append({
            "arxiv_id":         result.entry_id.split("/")[-1],
            "title":            result.title,
            "authors":          [a.name for a in result.authors],
            "abstract":         result.summary,
            "categories":       result.categories,
            "primary_category": result.primary_category,
            "published":        published,
            "updated":          result.updated.strftime("%Y-%m-%d"),
            "pdf_url":          result.pdf_url,
        })
        if len(papers) >= max_papers:
            break
    return papers


def download_pdf(paper, save_dir, max_retries):
    safe_id  = paper["arxiv_id"].replace("/", "_").replace(":", "_")
    pdf_path = save_dir / f"{safe_id}.pdf"
    if pdf_path.exists():
        return True
    for attempt in range(max_retries):
        try:
            r = requests.get(paper["pdf_url"], timeout=30)
            r.raise_for_status()
            pdf_path.write_bytes(r.content)
            return True
        except Exception:
            if attempt == max_retries - 1:
                return False
            time.sleep(2 ** attempt)
    return False


def main():
    args     = parse_args()
    pdf_dir  = Path(args.output_dir)
    meta_path = Path(args.metadata_path)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    all_papers = {}
    for category in args.categories:
        for paper in search_category(category, args.max_papers, args.start_date, args.end_date, args.rate_limit):
            all_papers.setdefault(paper["arxiv_id"], paper)
        time.sleep(args.rate_limit)

    papers = list(all_papers.values())
    with open(meta_path, "w") as f:
        json.dump(papers, f, indent=2)
    print(f"Metadata: {len(papers)} papers → {meta_path}")

    if not args.no_pdf:
        ok = failed = 0
        for paper in tqdm(papers, desc="Downloading PDFs"):
            if download_pdf(paper, pdf_dir, args.max_retries):
                ok += 1
            else:
                failed += 1
            time.sleep(args.rate_limit)
        print(f"PDFs — success: {ok} | failed: {failed}")


if __name__ == "__main__":
    main()