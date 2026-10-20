"""Inspectable Cedar Lab retrieval. Keyword path uses only the standard library."""
import argparse
from dataclasses import dataclass
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
STOPWORDS = set('a an the is are to of in at for and or may can i it be'.split())
QUESTIONS = {
    'Q1': 'May a trained graduate student operate the laser alone at 19:00?',
    'Q2': 'Can I use the laser without another person present?',
    'Q3': 'How frequently must I refresh my laser qualification?',
    'Q4': 'What is the laser wavelength?',
}
REQUIRED = {'Q1': {'A1', 'A2'}, 'Q2': {'A2'}, 'Q3': {'T1'}}

@dataclass(frozen=True)
class Chunk:
    id: str
    filename: str
    text: str


def load_chunks():
    """LOAD: separate labels from searchable paragraph text."""
    chunks = []
    for filename in ('access.txt', 'training.txt', 'booking.txt'):
        for paragraph in (ROOT / 'docs' / filename).read_text().strip().split('\n\n'):
            match = re.fullmatch(r'\[([ATB][12])\] (.+)', paragraph)
            if not match:
                raise ValueError(f'Invalid chunk in {filename}: {paragraph!r}')
            chunks.append(Chunk(match[1], filename, match[2]))
    if len(chunks) != 6 or {c.id for c in chunks} != {'A1','A2','T1','T2','B1','B2'}:
        raise ValueError('Expected exactly six uniquely labeled chunks')
    return chunks


def words(text):
    return set(re.findall(r'[a-z0-9]+', text.lower())) - STOPWORDS


def keyword_rank(chunks, question):
    query = words(question)
    # SCORE: each distinct shared word contributes exactly one point.
    rows = [(c, len(query & words(c.text)), sorted(query & words(c.text))) for c in chunks]
    # RANK: descending score, then ascending chunk ID.
    return sorted(rows, key=lambda r: (-r[1], r[0].id))


def print_ranking(rows, k, keyword=False):
    for c, score, matched in rows:
        value = str(score) if keyword else f'{score:.8f}'
        print(f'{c.id} | {c.filename} | score={value}' + (f' | matched={matched}' if keyword else ''))
    selected = ([r for r in rows if r[1] > 0] if keyword else rows)[:k]
    print('Selected: ' + ', '.join(r[0].id for r in selected) if selected else 'No matching passages.')
    for c, _, _ in selected:
        print(f'[{c.id}] ({c.filename}) {c.text}')
    return selected


class EmbeddingRetriever:
    def __init__(self, chunks):
        import numpy as np
        from sentence_transformers import SentenceTransformer
        self.np, self.chunks = np, chunks
        self.model_name = 'sentence-transformers/all-MiniLM-L6-v2'
        self.model = SentenceTransformer(self.model_name, device='cpu', cache_folder=str(ROOT / '.cache' / 'models'))
        # ENCODE DOCUMENTS ONCE, excluding IDs and filenames.
        self.matrix = self.model.encode_document([c.text for c in chunks], normalize_embeddings=True, convert_to_numpy=True)
        assert self.matrix.shape == (6, 384), self.matrix.shape
        assert np.allclose(np.linalg.norm(self.matrix, axis=1), 1, atol=1e-6)

    def rank(self, question):
        np = self.np
        vector = self.model.encode_query(question, normalize_embeddings=True, convert_to_numpy=True)
        assert vector.shape == (384,), vector.shape
        assert np.isclose(np.linalg.norm(vector), 1, atol=1e-6)
        scores = self.matrix @ vector  # All 384 coordinates, full stored precision.
        cosine = scores / (np.linalg.norm(self.matrix, axis=1) * np.linalg.norm(vector))
        assert np.allclose(scores, cosine, atol=1e-6)
        rows = sorted([(c, float(s), []) for c,s in zip(self.chunks, scores)], key=lambda r: (-r[1], r[0].id))
        return rows, vector

    def environment(self):
        import platform
        from importlib.metadata import version
        print(f'Model: {self.model_name}\nDevice: cpu\nPython: {platform.python_version()}')
        for package in ('sentence-transformers', 'numpy', 'torch', 'transformers', 'huggingface-hub'):
            print(f'{package}: {version(package)}')
        snapshots = ROOT / '.cache/models/models--sentence-transformers--all-MiniLM-L6-v2/snapshots'
        if snapshots.exists():
            print('Cached model revisions: ' + ', '.join(p.name for p in snapshots.iterdir()))
        print(f'Document matrix shape: {self.matrix.shape}')

    def inspect(self, question):
        np = self.np
        self.environment()
        print('All 384 coordinates are used; only the first 12 are displayed. Coordinates have no individual semantic labels.')
        for c,v in zip(self.chunks,self.matrix):
            print(f'{c.id} first12=' + ' '.join(f'{x:.4f}' for x in v[:12]) + f' length={np.linalg.norm(v):.8f}')
        rows,q = self.rank(question)
        print(f'Question: {question}\nQuestion vector shape: {q.shape}')
        print('Query first12=' + ' '.join(f'{x:.4f}' for x in q[:12]) + f' length={np.linalg.norm(q):.8f}')
        print_ranking(rows, 2)
        index = self.chunks.index(rows[0][0])
        d = self.matrix[index]
        products = q * d
        full = float(np.dot(q,d))
        direct = full / float(np.linalg.norm(q)*np.linalg.norm(d))
        print(f'Q1 top passage: {rows[0][0].id}')
        print('First12 products: ' + ' '.join(f'{x:.4f}' for x in products[:12]))
        print(f'First12 subtotal: {products[:12].sum():.8f}\nRemaining372 subtotal: {products[12:].sum():.8f}\nFull dot product: {full:.8f}\nDirect cosine: {direct:.8f}\nAgreement atol=1e-6: {np.isclose(full,direct,atol=1e-6)}')
        shown = [(c.id,i+1,float(x)) for c,v in zip(self.chunks,self.matrix) for i,x in enumerate(v[:12])]
        shown += [('query',i+1,float(x)) for i,x in enumerate(q[:12])]
        print(f'Displayed minimum (ID, coordinate, value): {min(shown,key=lambda x:x[2])}')
        print(f'Displayed maximum (ID, coordinate, value): {max(shown,key=lambda x:x[2])}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('method', choices=['keyword', 'embedding', 'inspect'])
    parser.add_argument('question', nargs='?', default=QUESTIONS['Q1'])
    parser.add_argument('-k', type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.k <= 6:
        parser.error('k must be between 1 and 6')
    chunks = load_chunks()
    if args.method == 'keyword':
        print_ranking(keyword_rank(chunks,args.question),args.k,keyword=True)
    else:
        retriever = EmbeddingRetriever(chunks)
        if args.method == 'inspect':
            retriever.inspect(args.question)
        else:
            retriever.environment()
            rows,_ = retriever.rank(args.question)
            print_ranking(rows,args.k)

if __name__ == '__main__':
    main()
