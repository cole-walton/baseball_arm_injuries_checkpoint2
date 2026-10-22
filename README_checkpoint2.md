# Baseball arm injuries — Checkpoint #2

This is a **starter Checkpoint #2 add-on** to merge with your own Checkpoint #1 project, not a replacement for its previous content.

## Files

- `checkpoint2_addon.ipynb`: only the new Checkpoint #2 cells, intended to copy into your existing `source.ipynb` **before its last nbconvert cell**.
- `source.ipynb`: a self-contained Checkpoint #2 template with the required final nbconvert cell; if used standalone, **add back your original Checkpoint #1 content**.
- `requirements.txt`: packages needed to run it.

## Instructions

1. Back up your existing `source.ipynb` in the Checkpoint #1 GitHub repository.
2. Either copy the sections from `checkpoint2_addon.ipynb` into your original notebook, or use the provided `source.ipynb` template and restore your earlier sections.
3. Install dependencies with `pip install -r requirements.txt`.
4. Run all notebook cells with internet access. The notebook tries the public Jon Roegele spreadsheet and then an archived GitHub dataset. If automatic download fails, download the **TJ List** tab as CSV and save it at `data/tj_surgeries.csv`.
5. Review the actual counts, graphs, missingness, and suspicious observations. Replace bracketed feedback/reflection placeholders with truthful comments and actual findings.
6. Ensure the very last cell of the final notebook is `!jupyter nbconvert --to python source.ipynb` and **run it**. Commit `source.ipynb`, resulting `source.py`, and the downloaded actual CSV to your existing GitHub repository.

## Key caution

The data contain documented surgery **cases**, not all athletes at risk. Charts show **counts**, not injury rates or causal effects. Documented return-time analyses are restricted to players with known same-level return outcomes. Never treat blank return times as zero or as proof the player did not return.

## Sources

- Jon Roegele, [Tommy John Surgery List](https://docs.google.com/spreadsheets/d/1gQujXQQGOVNaiuwSN680Hq-FDVsCwvN-3AazykOBON0/edit)
- [Archived dataset (GitHub)](https://github.com/derrickvasquez/Tommy-John-Surgery-Analysis)

Assignment due October 25, 2026; peer-review due November 1, 2026.
