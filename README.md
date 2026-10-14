# Dynamic Parsing & Dynamic Chunking for RAG

A small, dependency-free ingestion pipeline that demonstrates the ideas from the
"RAG: Dynamic Parsing & Dynamic Chunking" deck and compares chunking strategies.

## What it does

1. **Dynamic parsing** - detects the file type (Markdown, HTML, CSV), routes it to the
   right parser, and validates the result. Output is structured blocks with metadata
   (type, source, section), not one flat string.
2. **Dynamic chunking** - splits on structure (headings, table rows), merges small
   blocks up to a target token size, enforces a hard maximum, and prefixes each chunk
   with its section path. Table rows keep their column headers.
3. **Baseline** - fixed-size chunking that ignores structure.
4. **Evaluation** - BM25 retrieval and Recall@k over a small question set.

## Run

```bash
python rag_ingestion.py
```

Requires Python 3.8+. No third-party packages.

## Example result

```
fixed      chunks=5   Recall@2 = 4/5
dynamic    chunks=5   Recall@2 = 5/5
```

Fixed-size splitting separates the 15-minute expiry rule from the "Exceptions" section,
so one question fails. Structure-aware chunks keep each rule with its section.

## Use your own data

Edit `DOCS` (documents) and `EVAL_SET` (question, expected evidence text) in
`rag_ingestion.py`, then run again. Tune `TARGET_TOKENS` and `MAX_TOKENS`.

## Possible extensions

- PDF parsing and OCR routes
- Embedding-based retrieval instead of BM25
- Parent-child retrieval
- Per-content-type chunk size rules
