from pypdf import PdfReader
from pathlib import Path
import re

pdf_path = Path("docs/main.pdf")
reader = PdfReader(str(pdf_path))
print(f"pages={len(reader.pages)}")

all_text = []
for i, page in enumerate(reader.pages):
    txt = page.extract_text() or ""
    txt = txt.replace("\r", "\n")
    all_text.append(txt)

full = "\n\n".join(all_text)

# Print likely section lines and equation-heavy lines
lines = [ln.strip() for ln in full.splitlines() if ln.strip()]
sec_hits = []
eq_hits = []
keywords = [
    "assumption", "theorem", "proposition", "lemma", "equilibrium", "profit",
    "loss", "demand", "price", "outside option", "stackelberg", "sensitivity",
    "alpha", "beta", "gamma", "rho", "tau", "k", "c_T", "a", "b"
]
for ln in lines:
    low = ln.lower()
    if re.match(r"^(\d+\.|[ivx]+\.|chapter|section)", low):
        sec_hits.append(ln)
    if any(k in low for k in keywords):
        if len(eq_hits) < 220:
            eq_hits.append(ln)

print("\n=== SECTION CANDIDATES (first 80) ===")
for ln in sec_hits[:80]:
    print(ln)

print("\n=== KEY LINES (first 220) ===")
for ln in eq_hits[:220]:
    print(ln)

# Also dump first 3 pages for manual scan
print("\n=== FIRST 3 PAGES RAW TEXT ===")
for i in range(min(3, len(reader.pages))):
    print(f"\n--- PAGE {i+1} ---")
    print((all_text[i] or "")[:4000])
