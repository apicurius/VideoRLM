#!/bin/bash
cd /home/berkin/Documents/academic_research/VideoRLM
source .venv/bin/activate

echo "=== LanguageBind Integration Verification ==="
echo ""

# Test 1: Imports
echo -n "1. Embedder factory imports: "
timeout 30 python3 -c "
from kuavi.embedders import create_embedder, BaseEmbedder
e = create_embedder()
print('PASS:', type(e).__name__)
" 2>/dev/null || echo "FAIL"

# Test 2: Backend env switch
echo -n "2. Backend env switch: "
timeout 30 env EMBEDDING_BACKEND=languagebind python3 -c "
from kuavi.embedders import create_embedder
e = create_embedder()
assert 'LanguageBind' in type(e).__name__
print('PASS')
" 2>/dev/null || echo "FAIL"

# Test 3: Text similarity
echo -n "3. Text similarity: "
timeout 30 python3 -c "
from kuavi.embedders import create_embedder
e = create_embedder()
sim = e.similarity(
    e.embed_query('person picks up object'),
    e.embed_text('someone lifts a box')
)
status = 'PASS' if sim > 0.5 else 'FAIL'
print(f'{status}: {sim:.3f} (need > 0.5)')
" 2>/dev/null || echo "FAIL"

# Test 4: Video embedding (if video exists)
echo -n "4. Video embedding: "
if [ -f "big_bang_theory.mp4" ]; then
  timeout 60 python3 -c "
from kuavi.embedders import create_embedder
e = create_embedder()
emb = e.embed_video_segment('big_bang_theory.mp4', 0.0, 10.0)
status = 'PASS' if len(emb) > 100 else 'FAIL'
print(f'{status}: dim={len(emb)}')
" 2>/dev/null || echo "FAIL"
else
  echo "SKIP (video not found)"
fi

# Test 5: SigLIP fully removed
echo -n "5. SigLIP removed: "
result=$(grep -r "siglip\|SigLIP" kuavi/ --include="*.py" 2>/dev/null | head -1)
[ -z "$result" ] && echo "PASS" || echo "FAIL: $result"

# Test 6: Gemma embeddings fully removed
echo -n "6. Gemma embeddings removed: "
result=$(grep -r "SentenceTransformer\|gemma.*embed" kuavi/ --include="*.py" 2>/dev/null | head -1)
[ -z "$result" ] && echo "PASS" || echo "FAIL: $result"

# Test 7: Indexing pipeline has languagebind stage
echo -n "7. Indexer has languagebind stage: "
grep -q "languagebind\|LanguageBind" kuavi/indexer.py && \
  echo "PASS" || \
  echo "FAIL"

# Test 8: Tier 2 uses new embedder
echo -n "8. Tier 2 uses new embedder: "
grep -q "embed_query\|create_embedder\|LanguageBind" kuavi/tier_executors.py && \
  echo "PASS" || \
  echo "FAIL"

# Test 9: Pytest
echo -n "9. Pytest: "
timeout 120 python3 -m pytest tests/ -x -q --tb=line 2>&1 | tail -3

echo ""
echo "=== Verification Complete ==="
