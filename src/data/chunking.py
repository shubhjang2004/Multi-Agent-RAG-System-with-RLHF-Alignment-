import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import fitz
import nltk
import spacy
from tqdm import tqdm

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nlp = spacy.load("en_core_web_sm")


def parse_args():
    parser = argparse.ArgumentParser(description="Chunk ArXiv PDFs")
    parser.add_argument("--pdf_dir",       type=str, default="data/raw/arxiv_papers")
    parser.add_argument("--metadata",      type=str, default="data/raw/arxiv_metadata.json")
    parser.add_argument("--output_dir",    type=str, default="data/processed/chunks")
    parser.add_argument("--tracker",       type=str, default="data/chunk_tracker.json")
    parser.add_argument("--chunk_size",    type=int, default=512)
    parser.add_argument("--chunk_overlap", type=int, default=128)
    parser.add_argument("--min_words",     type=int, default=20)
    parser.add_argument("--batch_size",    type=int, default=1000)
    parser.add_argument("--save_every",    type=int, default=100)
    return parser.parse_args()


def load_tracker(tracker_path, all_ids):
    if Path(tracker_path).exists():
        with open(tracker_path) as f:
            tracker = json.load(f)
        known = set(tracker["chunked"] + tracker["pending"] + tracker["failed"])
        new   = [p for p in all_ids if p not in known]
        if new:
            tracker["pending"].extend(new)
        return tracker
    tracker = {"chunked": [], "pending": all_ids, "failed": [], "total": len(all_ids), "updated_at": str(datetime.now())}
    save_tracker(tracker_path, tracker)
    return tracker


def save_tracker(tracker_path, tracker):
    tracker["updated_at"] = str(datetime.now())
    with open(tracker_path, "w") as f:
        json.dump(tracker, f, indent=2)


def extract_text(pdf_path):
    try:
        doc  = fitz.open(pdf_path)
        text = "".join(page.get_text() for page in doc)
        doc.close()
        return text
    except Exception:
        return None


def clean_text(text):
    text = re.sub(r"\s+",     " ",    text)
    text = re.sub(r"\n\d+\n", "\n",   text)
    text = re.sub(r"\n{3,}",  "\n\n", text)
    text = re.sub(r"http\S+", "",     text)
    return text.strip()


def remove_references(text):
    for pat in [r"\nReferences\s*\n", r"\nREFERENCES\s*\n", r"\nBibliography\s*\n"]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return text[:m.start()]
    return text


def is_good_chunk(chunk, min_words):
    words = chunk.split()
    if len(words) < min_words:
        return False
    num_ratio     = sum(1 for w in words if w.replace(".", "").replace(",", "").isdigit()) / len(words)
    special_ratio = sum(1 for c in chunk if not c.isalnum() and not c.isspace()) / len(chunk)
    return num_ratio <= 0.5 and special_ratio <= 0.3


def semantic_chunks(text, chunk_size, overlap):
    sentences = [s.text.strip() for s in nlp(text).sents]
    chunks, current, cur_len = [], [], 0
    for sent in sentences:
        sent_len = len(sent.split())
        if cur_len + sent_len > chunk_size and current:
            chunks.append(" ".join(current))
            ov, ov_len = [], 0
            for s in reversed(current):
                l = len(s.split())
                if ov_len + l <= overlap:
                    ov.insert(0, s); ov_len += l
                else:
                    break
            current, cur_len = ov, ov_len
        current.append(sent)
        cur_len += sent_len
    if current:
        chunks.append(" ".join(current))
    return chunks


def process_paper(pdf_path, metadata, args):
    text = extract_text(pdf_path)
    if not text:
        return []
    text = clean_text(text)
    text = remove_references(text)
    return [
        {
            "chunk_id":    f"{metadata['arxiv_id']}_{i}",
            "text":        chunk,
            "arxiv_id":    metadata["arxiv_id"],
            "title":       metadata["title"],
            "authors":     metadata["authors"],
            "abstract":    metadata["abstract"],
            "categories":  metadata["categories"],
            "published":   metadata["published"],
            "chunk_index": i,
            "word_count":  len(chunk.split()),
        }
        for i, chunk in enumerate(semantic_chunks(text, args.chunk_size, args.chunk_overlap))
        if is_good_chunk(chunk, args.min_words)
    ]


def main():
    args       = parse_args()
    pdf_dir    = Path(args.pdf_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.metadata) as f:
        papers = json.load(f)
    metadata_lookup = {}
    for p in papers:
        metadata_lookup[p["arxiv_id"]]                   = p
        metadata_lookup[p["arxiv_id"].replace("/", "_")] = p

    pdf_files   = {f.stem: f for f in pdf_dir.glob("*.pdf")}
    tracker     = load_tracker(args.tracker, list(pdf_files.keys()))
    batch       = tracker["pending"][:args.batch_size]
    batch_index = len(list(output_dir.glob("chunks_batch_*.json"))) + 1

    print(f"Chunked: {len(tracker['chunked'])} | Pending: {len(tracker['pending'])} | Processing: {len(batch)}")

    buffer = []
    ok = failed = 0

    for i, paper_id in enumerate(tqdm(batch, desc="Chunking")):
        
        pdf_path = pdf_files.get(paper_id)
        metadata = metadata_lookup.get(paper_id)
        print(f"{paper_id} | pdf: {pdf_path} | meta: {metadata is not None}")
        text = extract_text(pdf_path)
        print(f"  text length: {len(text) if text else 'None'}")

        if not pdf_path or not metadata:
            tracker["pending"].remove(paper_id)
            tracker["failed"].append(paper_id)
            failed += 1
            continue

        chunks = process_paper(pdf_path, metadata, args)
        tracker["pending"].remove(paper_id)

        if chunks:
            buffer.extend(chunks)
            tracker["chunked"].append(paper_id)
            ok += 1
        else:
            tracker["failed"].append(paper_id)
            failed += 1

        if (i + 1) % args.save_every == 0:
            out = output_dir / f"chunks_batch_{batch_index}.json"
            with open(out, "w") as f:
                json.dump(buffer, f, indent=2)
            save_tracker(args.tracker, tracker)
            buffer      = []
            batch_index += 1

    if buffer:
        out = output_dir / f"chunks_batch_{batch_index}.json"
        with open(out, "w") as f:
            json.dump(buffer, f, indent=2)

    save_tracker(args.tracker, tracker)
    print(f"Done — success: {ok} | failed: {failed} | total chunked: {len(tracker['chunked'])}/{tracker['total']}")


if __name__ == "__main__":
    main()