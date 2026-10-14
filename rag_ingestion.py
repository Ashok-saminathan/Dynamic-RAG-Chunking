"""Small RAG ingestion pipeline: dynamic parsing + dynamic chunking + evaluation.

Run: python3 rag_ingestion.py
Pure standard library, no external dependencies.
"""
import csv
import io
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from html.parser import HTMLParser

TARGET_TOKENS = 60   # small values so the demo corpus shows the effect
MAX_TOKENS = 90


# ---------------------------------------------------------------- data model
@dataclass
class Block:
    text: str
    type: str                  # paragraph | heading | table_row | code
    source: str
    section: str = ""
    meta: dict = field(default_factory=dict)


@dataclass
class Chunk:
    text: str
    source: str
    section: str
    content_type: str
    chunk_id: str = ""


def count_tokens(text):
    return len(text.split())


# ------------------------------------------------------------ dynamic parsing
class _HTMLText(HTMLParser):
    SKIP = {"nav", "footer", "script", "style", "header"}

    def __init__(self):
        super().__init__()
        self.depth_skip = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.depth_skip += 1
        elif tag in ("h1", "h2", "h3"):
            self.parts.append("\n# ")
        elif tag in ("p", "li", "br"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self.depth_skip -= 1

    def handle_data(self, data):
        if self.depth_skip == 0 and data.strip():
            self.parts.append(data.strip() + " ")


def parse_markdown(text, source):
    blocks, section, buf = [], "", []

    def flush():
        if buf:
            blocks.append(Block(" ".join(buf).strip(), "paragraph", source, section))
            buf.clear()

    for line in text.splitlines():
        if line.startswith("#"):
            flush()
            section = line.lstrip("# ").strip()
            blocks.append(Block(section, "heading", source, section))
        elif not line.strip():
            flush()
        else:
            buf.append(line.strip())
    flush()
    return blocks


def parse_csv(text, source):
    rows = list(csv.DictReader(io.StringIO(text)))
    return [
        Block(" | ".join(f"{k}: {v}" for k, v in row.items()),
              "table_row", source, "table", {"row": i})
        for i, row in enumerate(rows)
    ]


def parse_html(text, source):
    p = _HTMLText()
    p.feed(text)
    return parse_markdown("".join(p.parts), source)


def detect_format(name, content):
    """Detect: decide the route from the extension and the observed content."""
    ext = name.rsplit(".", 1)[-1].lower()
    if ext in ("md", "txt") and "<html" in content.lower():
        return "html"
    return {"md": "markdown", "txt": "markdown", "html": "html",
            "csv": "csv"}.get(ext, "unknown")


def parse_document(name, content):
    """Route -> Validate. Returns structured blocks, never one flat string."""
    fmt = detect_format(name, content)
    router = {"markdown": parse_markdown, "html": parse_html, "csv": parse_csv}
    if fmt not in router:
        raise ValueError(f"No parser for {name}; flag for review, do not index")
    blocks = router[fmt](content, name)
    if not blocks:
        raise ValueError(f"Empty extraction for {name}; flag for review")
    return blocks


# ----------------------------------------------------------- dynamic chunking
def split_oversized(text, max_tokens):
    """Split at sentence boundaries first; hard-cut only as a last resort."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    out, cur = [], []
    for s in sentences:
        if count_tokens(" ".join(cur + [s])) > max_tokens and cur:
            out.append(" ".join(cur))
            cur = []
        cur.append(s)
        while count_tokens(" ".join(cur)) > max_tokens:  # one huge sentence
            words = " ".join(cur).split()
            out.append(" ".join(words[:max_tokens]))
            cur = [" ".join(words[max_tokens:])]
    if cur:
        out.append(" ".join(cur))
    return out


def dynamic_chunk(blocks):
    chunks, buf, buf_section, buf_type = [], [], None, None

    def flush():
        nonlocal buf
        if buf:
            chunks.append(Chunk(" ".join(buf), blocks[0].source,
                                buf_section, buf_type))
            buf = []

    for b in blocks:
        if b.type == "heading":
            flush()              # structure boundary: never merge across sections
            continue
        # tables: each row stands alone with its headers already attached
        if b.type == "table_row":
            flush()
            chunks.append(Chunk(b.text, b.source, b.section, "table_row"))
            continue
        # prose: prefix the section path so context travels with the chunk
        text = f"[{b.section}] {b.text}" if b.section else b.text
        for unit in split_oversized(text, MAX_TOKENS):
            same = buf_section == b.section and buf_type == b.type
            if buf and (not same or count_tokens(" ".join(buf + [unit])) > TARGET_TOKENS):
                flush()
            buf_section, buf_type = b.section, b.type
            buf.append(unit)
    flush()
    for i, c in enumerate(chunks):
        c.chunk_id = f"{c.source}#{i}"
    return chunks


def fixed_chunk(blocks, size=40):
    """Baseline: flat text split every N tokens, structure ignored."""
    source = blocks[0].source
    words = " ".join(b.text for b in blocks).split()
    return [Chunk(" ".join(words[i:i + size]), source, "", "flat", f"{source}#{i // size}")
            for i in range(0, len(words), size)]


# ------------------------------------------------------------ retrieval (BM25)
class BM25:
    def __init__(self, chunks, k1=1.5, b=0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        self.docs = [Counter(self._tok(c.text)) for c in chunks]
        self.lens = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.lens) / len(self.lens)
        df = Counter(t for d in self.docs for t in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    @staticmethod
    def _tok(text):
        return re.findall(r"[a-z0-9]+", text.lower())

    def search(self, query, k=3):
        q = self._tok(query)
        scores = []
        for i, d in enumerate(self.docs):
            s = 0.0
            for t in q:
                if t in d:
                    f = d[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (
                        f + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg))
            scores.append(s)
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [self.chunks[i] for i in order]


# --------------------------------------------------------------- demo corpus
POLICY = """# Password Reset
Select "Forgot password" on the login page. A link is sent to the verified work email and expires after 15 minutes.
Use the link only once. Choose a password with at least 12 characters and do not reuse an old password.

# Exceptions
If the employee cannot access the work email, the identity team must verify them using the approved recovery workflow.
The identity team will ask for a manager approval before issuing a temporary credential.

# Leave Policy
Employees receive 24 days of paid leave per year. Unused leave can be carried forward for up to 6 months.
"""

PLANS = """plan,monthly_cost,coverage_limit
Plan A,500,10000
Plan B,900,20000
"""

DOCS = {"policy.md": POLICY, "plans.csv": PLANS}

# (question, text the retrieved evidence must contain)
EVAL_SET = [
    ("how long does the reset link last", "expires after 15 minutes"),
    ("what if employee cannot access work email", "identity team must verify"),
    ("monthly cost of Plan B", "monthly_cost: 900"),
    ("coverage limit of Plan A", "coverage_limit: 10000"),
    ("how many days of paid leave", "24 days"),
]


def evaluate(name, chunker, k=2):
    chunks = []
    for fname, content in DOCS.items():
        chunks += chunker(parse_document(fname, content))
    index = BM25(chunks)
    hits = 0
    for question, evidence in EVAL_SET:
        top = index.search(question, k)
        hits += any(evidence in c.text for c in top)
    print(f"{name:<10} chunks={len(chunks):<3} Recall@{k} = {hits}/{len(EVAL_SET)}")
    return chunks


if __name__ == "__main__":
    evaluate("fixed", lambda blocks: fixed_chunk(blocks, size=25))
    dynamic = evaluate("dynamic", dynamic_chunk)
    print("\nSample dynamic chunks:")
    for c in dynamic:
        print(f"- {c.chunk_id} | {c.content_type} | {c.section}\n    {c.text}")
