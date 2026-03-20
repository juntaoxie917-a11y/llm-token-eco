from pathlib import Path
import traceback

out = Path('tmp/main_pdf_extract.txt')
try:
    from PyPDF2 import PdfReader
    reader = PdfReader('docs/main.pdf')
    chunks = []
    chunks.append(f'pages={len(reader.pages)}\n')

    full_lines = []
    for i, page in enumerate(reader.pages):
        txt = (page.extract_text() or '').replace('\r', '\n')
        full_lines.extend([ln.strip() for ln in txt.splitlines() if ln.strip()])
        if i < 3:
            chunks.append(f'\n--- PAGE {i+1} ---\n')
            chunks.append(txt[:5000])
            chunks.append('\n')

    keys = ['assumption','theorem','proposition','lemma','equilibrium','profit','loss','demand','price','outside option','stackelberg','sensitivity','alpha','beta','gamma','rho','tau','k','c_T']
    hits = []
    for ln in full_lines:
        low = ln.lower()
        if any(k in low for k in keys):
            hits.append(ln)
            if len(hits) >= 400:
                break

    chunks.append('\n=== KEY LINES ===\n')
    chunks.extend([h + '\n' for h in hits])
    out.write_text(''.join(chunks), encoding='utf-8')
except Exception:
    out.write_text('ERROR\n' + traceback.format_exc(), encoding='utf-8')
