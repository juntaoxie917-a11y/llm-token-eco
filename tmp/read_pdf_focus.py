from pypdf import PdfReader

reader = PdfReader('docs/main.pdf')
lines = []
for page in reader.pages:
    txt = (page.extract_text() or '').replace('\r','\n')
    lines.extend([ln.strip() for ln in txt.splitlines() if ln.strip()])

targets = [
    'Sensitivity Analysis', 'parameter range', 'outside option', 'soft outside option',
    'student payoff', 'teacher payoff', 'best-response', 'D*', 'p*',
    'alpha', 'beta', 'gamma', 'rho', 'q', 'k', 'c_T', 'a', 'b', 'tau'
]

# print windows around key section markers
for i, ln in enumerate(lines):
    low = ln.lower()
    if any(t.lower() in low for t in ['sensitivity analysis', 'outside option mechanism', 'soft outside option', 'student payoff', 'teacher payoff']):
        print('\n' + '='*30)
        print(f'LINE {i+1}: {ln}')
        start = max(0, i-4)
        end = min(len(lines), i+8)
        for j in range(start, end):
            print(f'{j+1}: {lines[j]}')

print('\n' + '='*30)
print('TARGET-HIT LIST (first 220)')
count = 0
for i, ln in enumerate(lines):
    low = ln.lower()
    if any(t.lower() in low for t in targets):
        print(f'{i+1}: {ln}')
        count += 1
    if count >= 220:
        break
