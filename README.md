# Cedar Lab RAG Challenge

PS05 compares exact keyword overlap with real local MiniLM embeddings over six fictional laboratory rules. Python retrieves evidence; the final answer stage is a manual handoff to fresh Codex conversations. This is a teaching workflow, not a deployed chatbot.

## Status and authors

- Assistance disclosure: Codex helped implement the code and prepare audits. The reflection below expresses Rey's observations and proposed experiment, with English wording edited by Codex.
- Student: **Rey**. Listed author: Rey; no partner has been listed.
- Evidence judgments and rationale: `results.csv` and `evidence/retrieval-audit.txt`.
- Q1 original/revised answers, Q4 answer, and claim audits: saved in `evidence/`. The Q1 instruction revision and before/after comparison are included.
- Individual reflection: included below (under 150 words).

## Setup and commands

From this repository directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python retriever.py keyword "May a trained graduate student operate the laser alone at 19:00?" -k 2
python retriever.py embedding "May a trained graduate student operate the laser alone at 19:00?" -k 2
python retriever.py inspect > evidence/embedding-inspection.txt
python run_experiments.py
```

The keyword command needs only Python's standard library. The embedding commands need the installed packages and the first model download needs internet. Use CPU; no model API key is needed. Versions and the cached model revision are recorded in `evidence/versions.txt` after a successful run; `requirements-lock.txt` records the installed environment. The local environment used Python 3.14.8 on macOS. Dependency locks may need adaptation on another OS/Python version.

Model: `sentence-transformers/all-MiniLM-L6-v2`. Documents are encoded with `encode_document()`, questions with `encode_query()`, both with normalization and numpy output. The six document vectors are computed once per experiment process and reused across questions. IDs and filenames remain metadata and are never encoded. Model files are cached under `.cache/models/` and ignored by Git.

## Read the code

In `retriever.py`:

- `load_chunks()` (lines 24–35) reads the three files, separates labels from text, and validates exactly six distinct IDs.
- `words()` lowercases, extracts alphanumeric words, removes exactly the specified stopwords, and deduplicates using a set.
- `keyword_rank()` (lines 42–47) counts the intersection of query and document word sets. Sorting uses negative score followed by ID, giving descending scores and ascending IDs. Selection uses only positive scores.
- `EmbeddingRetriever.__init__()` (lines 62–71) loads the pretrained model on CPU and produces the document matrix `(6, 384)`. This runs the existing model; it does not train it.
- `rank()` (lines 73–82) makes a question vector `(384,)`, computes all six dot products with the document matrix, checks cosine agreement, sorts, and returns the ranked records. Embedding selection always takes top-k with no threshold.
- `inspect()` displays the first 12 coordinates, vector lengths, Q1 similarity decomposition and displayed extrema. All 384 coordinates contribute to retrieval; individual coordinates do not have reliable semantic names.

For verified line numbers on your checkout, run `nl -ba retriever.py`. `run_experiments.py` saves actual rankings and prepares Task 4 prompts; it does not generate answers or fill human judgment columns.

## Evidence and what to audit

- `evidence/pre-run-prediction.txt`: Q1 evidence target recorded before the first retrieval run.
- `evidence/keyword-Q1.txt`: actual keyword baseline output.
- `evidence/baseline-checks.txt`: source/scoring validation. `check_baseline.py` compares documents directly against the original handout located beside this repository; it is an optional local check, not needed by the retriever on a clean clone.
- `evidence/embedding-inspection.txt`: real vectors, shapes, lengths, and similarity calculation.
- `evidence/retrieval-runs.txt`: four questions through both methods at k=2.
- `evidence/k-experiment.txt`: Q1 at k=1,2,3, selected passages and required evidence coverage.
- `evidence/rankings.json`: full stored scores for selected passages.
- `results.csv`: four comparison rows with completed evidence judgments and observations.
- `handoff/Q1-prompt.txt`, `handoff/Q4-prompt.txt`: original answer prompts.
- `handoff/Q1-prompt-revised.txt`: revised Q1 prompt, keeping the same question and passages.
- `evidence/Q1-revision-comparison.txt`: instruction change, before/after results, and assessment.
- `evidence/Q1-answer-original.txt`, `evidence/Q1-answer-revised.txt`, `evidence/Q4-answer-original.txt`: preserved answer outputs.
- `evidence/Q1-audit.csv`, `evidence/Q1-audit-revised.csv`, `evidence/Q4-audit.csv`: claim-level audits.

Q1 needs A1 and A2 for complete evidence, Q2 needs A2, Q3 needs T1. Q4 has no wavelength in the corpus. A positive similarity is not proof that an answer exists. Keyword scores count words; embedding scores measure vector alignment, so their numeric scales cannot be compared directly.

For a manual keyword check, Q1 and A1 share `00`, `graduate`, and `trained`: score 3. Repeating a query word does not add points. The `00` match illustrates a lexical artifact: 19:00 and the access times produce the same token. `student` and `students` remain different words. Q1 keyword selection is A1 then A2.

## Completion and submission status

The source documents, retrievers, real vector inspection, four-question comparison, k experiment, original and revised Q1 answers, Q4 answer, and claim audits are saved. See `evidence/answer-audit-summary.txt` for the audit outcomes.

The original Q1 response omits explicit acknowledgement of missing access-hours evidence. Adding an instruction to check the access-hours condition at 19:00 fixes that omission in the revised response, using the same A2/T1 passages. Both versions, the revised prompt, and a before/after comparison are retained. The instruction improves answer disclosure, while retrieval coverage remains 1/2. Q4 correctly reports that the wavelength is unspecified in its supplied evidence.

The individual reflection below covers the question-to-cited-answer path, an observed limitation, and a proposed experiment.

The public [GitHub repository](https://github.com/reybao/cedar-rag) has been submitted to Sheet1!R16 (Rey Bao, PS5 Repo) on October 20, 2026; details are tracked in `SUBMISSION.md`. The official [course Sign Up Sheet](https://docs.google.com/spreadsheets/d/1ZewIG5udWWpk3Kdrab5yDbsWDV4gnH3liKobIAp7HG4/edit?usp=sharing) was extracted from the assignment's actual hyperlink.

No retrieval output is borrowed. Real model outputs are saved in the evidence files. Re-running `run_experiments.py` rewrites generated experiment files and resets CSV judgment columns, so preserve completed annotations before rerunning.

## References

- [Sentence Transformers API](https://www.sbert.net/docs/package_reference/sentence_transformer/model.html)
- [Model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- Assignment: PS05 and Lesson 10 RAG Systems handout supplied in the parent course folder.

## Actual run summary for orientation

The saved run selected the following top-two IDs (read the full logs and make your own judgments):

| Question | Keyword | Embedding |
| --- | --- | --- |
| Q1 | A1, A2 | A2, T1 |
| Q2 | A2, B1 | A2, T1 |
| Q3 | T1, A2 | T1, B1 |
| Q4 | A2, B1 | T1, A2 |

For Q1 the embedding ranking places A1 fourth, so k=1,2,3 all cover only one of the two required chunks. Moving from k=1 to 2 adds T1's training condition; moving to 3 adds B1's booking rule. Neither adds the missing access-hours rule. In the current run the displayed coordinate extrema are T1 coordinate 5 (minimum) and T1 coordinate 12 (maximum); neither coordinate has a standalone meaning.

## Individual reflection — Rey

This project showed me how retrieved passages become evidence for a cited answer. Keyword matching achieved better required-evidence coverage in this experiment: it retrieved both A1 and A2 for Q1, while embedding retrieval missed A1 at k=2 and k=3. This surprised me. A2 supported the conclusion that solo operation was prohibited, but the retrieved evidence lacked the schedule needed to explain access at 19:00. I learned that a correct conclusion does not necessarily mean complete evidence coverage. RAG can help reduce hallucinations, but retrieved passages may be incomplete and answers still require auditing. Next, I would combine normalized keyword and embedding scores with different weights. I would check whether this retrieves both A1 and A2 at k=2 while retaining the required evidence for Q2 and Q3.
