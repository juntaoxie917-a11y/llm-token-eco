from pathlib import Path
import importlib.util
pdf = Path('docs/main.pdf')
print('exists', pdf.exists())
mods = ['pypdf', 'PyPDF2', 'fitz']
print({m: bool(importlib.util.find_spec(m)) for m in mods})
reader = None
if importlib.util.find_spec('pypdf'):
    from pypdf import PdfReader
    reader = PdfReader(str(pdf))
elif importlib.util.find_spec('PyPDF2'):
    from PyPDF2 import PdfReader
    reader = PdfReader(str(pdf))
if reader is None:
    print('NO_READER')
    raise SystemExit(0)
print('pages', len(reader.pages))
text_parts = []
for i, p in enumerate(reader.pages[:10], start=1):
    t = p.extract_text() or ''
    text_parts.append(f'\n\n---PAGE {i}---\n' + t[:2000])
out = ''.join(text_parts)
print(out[:16000])
