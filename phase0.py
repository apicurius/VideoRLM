import os
for f in [
    'kuavi/embedders/__init__.py',
    'kuavi/embedders/gemini_embedder.py',
    'kuavi/indexer.py',
    'kuavi/search.py',
    'kuavi/context.py',
    'kuavi/tier_executors.py',
]:
    print(f'{"EXISTS" if os.path.exists(f) else "MISSING":8} {f}')
