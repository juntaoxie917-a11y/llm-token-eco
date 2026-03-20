from pathlib import Path
out = Path('tmp/check_pypdf.txt')
try:
    import pypdf
    out.write_text('OK ' + getattr(pypdf, '__version__', 'unknown'), encoding='utf-8')
except Exception as e:
    out.write_text('ERROR ' + repr(e), encoding='utf-8')
