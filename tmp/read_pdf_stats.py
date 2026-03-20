from pathlib import Path
from collections import Counter
from pypdf import PdfReader

pdf = Path('docs/main.pdf')
print('exists', pdf.exists())
reader = PdfReader(str(pdf))
print('pages', len(reader.pages))

lines = []
for page in reader.pages:
    txt = (page.extract_text() or '').replace('\r', '\n')
    for ln in txt.splitlines():
        ln = ln.strip()
        if ln:
            lines.append(ln)

keys = [
    'alpha', 'beta', 'gamma', 'rho', 'tau',
    'outside option', 'stackelberg', 'demand', 'price', 'profit', 'sensitivity'
]
counts = Counter()
for ln in lines:
    low = ln.lower()
    for k in keys:
        if k in low:
            counts[k] += 1

print('KEY_COUNTS_START')
for k in keys:
    print(k, counts[k])
print('KEY_COUNTS_END')

print('FIRST_PAGE_HEAD_START')
for ln in lines[:80]:
    print(ln)
print('FIRST_PAGE_HEAD_END')

print('KEY_HITS_START')
hit = 0
for i, ln in enumerate(lines, 1):
    low = ln.lower()
    if any(k in low for k in ['sensitivity', 'comparative statics', 'outside option', 'stackelberg', 'alpha', 'beta', 'gamma', 'rho', 'tau', 'demand', 'price', 'profit']):
        print(f'{i}: {ln}')
        hit += 1
    if hit >= 160:
        break
print('KEY_HITS_END')
