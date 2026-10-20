"""Check specified scoring, ties, and verbatim source preservation."""
import zipfile
import xml.etree.ElementTree as ET
from retriever import ROOT, QUESTIONS, load_chunks, keyword_rank, words
chunks=load_chunks()
handout=ROOT.parent/'ps05'/'PS05 and Lesson 10 — RAG Systems.docx'
with zipfile.ZipFile(handout) as z:
    root=ET.fromstring(z.read('word/document.xml'))
    paragraphs=[''.join(n.text or '' for n in p.findall('.//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t')) for p in root.findall('.//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p')]
for c in chunks: assert f'[{c.id}] {c.text}' in paragraphs, c.id
rank=keyword_rank(chunks,QUESTIONS['Q1'])
assert [(c.id,s) for c,s,_ in rank]==[('A1',3),('A2',2),('B1',1),('T1',1),('T2',1),('B2',0)]
assert keyword_rank(chunks,'laser laser laser')==keyword_rank(chunks,'laser')
assert all(s==0 for _,s,_ in keyword_rank(chunks,'zzzzzz'))
assert words('May I be at the lab?')=={'lab'}
print('PASS: six unique chunks verbatim against handout; Q1 scores; ID tie-break; repeated words; zero matches; stopwords.')
