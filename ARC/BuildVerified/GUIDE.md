# Filling the three open fields

## A. Kaggle notebook link
1. kaggle.com > Code > New Notebook > File > Import Notebook > upload `kaggle_notebook.ipynb`.
2. Right panel > Input > Add Input > search "ARC Prize 2026 ARC-AGI-2" (the competition) and add it. (No data attached: set Internet = On; the notebook then clones the public data.)
3. Run All. Expect "PUBLIC EVALUATION: 0/167 ..." and a `submission.json` in Output.
4. Save Version > Save & Run All (Commit). Then Share > Public.
5. Copy the notebook URL into writeup section 6 table and section 10.

## B. Kaggle code submission
The competition closed on 2 Nov 2026. Open the competition > Submissions: if you already have a submission, link it. Otherwise try "Submit to Competition" from the notebook output; if it is refused, write "not submitted (competition closed)" in the table. Do not leave a placeholder.

## C. Zenodo DOI for the code (optional, 10 minutes)
1. zenodo.org > log in with GitHub > profile > GitHub > flip the switch for Sivatech24/Project-Intelligence.
2. On GitHub: Releases > Draft a new release > tag v0.1.0 > Publish. (Commit `CITATION.cff` first.)
3. Zenodo builds the archive; copy the DOI (10.5281/zenodo.NNNNNNN) into writeup section 11 and `CITATION.cff` (`doi:` field).

## D. Recursive-core notebook (new)
`kaggle_notebook_recursive.ipynb` implements the Target v3 diagram (recursive core, deep supervision, halt head, external verifier, selector, write gate) and prints the table and two charts. Set `N_TASKS = 15` for a 15-minute run, 60 for about an hour on 2 CPU cores. Measured: results_trm_training.json.
