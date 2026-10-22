#!/usr/bin/env python
# coding: utf-8

# # Baseball Arm Injuries: Tommy John Surgery Trends and Recovery
# 
# **Final Project — Checkpoint #2**  
# **Topic:** Arm injuries in baseball, focused on Tommy John surgery (ulnar collateral ligament reconstruction).
# 
# ### Research question
# How have **documented Tommy John surgeries** changed over time, and how do **age, position, throwing hand, and level of play** relate to the characteristics and documented recovery times of these cases?
# 
# > **Scope:** This dataset contains *surgery cases*, not every baseball player. It can describe reported surgeries and recovery among documented cases, but cannot estimate injury *risk* or prove which characteristics *cause* injuries without a comparison population.
# 
# **Project continuity:** Keep the Checkpoint #1 problem statement and dataset discussion in your existing `source.ipynb`. If your original research question differs, revise this wording to match your earlier work. The sections below fulfill Checkpoint #2.

# ## Data Sources
# 
# **Primary source:** [Jon Roegele's Tommy John Surgery List](https://docs.google.com/spreadsheets/d/1gQujXQQGOVNaiuwSN680Hq-FDVsCwvN-3AazykOBON0/edit). This public spreadsheet documents reported surgeries and fields including player, surgery date, competition level, position, handedness, age, and return/recovery information where known.
# 
# **Backup snapshot:** [Archived Tommy John surgery CSV on GitHub](https://github.com/derrickvasquez/Tommy-John-Surgery-Analysis/blob/Main/Updated%20Tommy%20John%20Surgery%20List%20%28%40MLBPlayerAnalys%29.csv). The backup may be older than the live list.
# 
# The notebook downloads the primary dataset the first time you run it, or uses a copy at `data/tj_surgeries.csv` if one already exists. **Commit that actual CSV file to your GitHub repository** so peer reviewers can reproduce the analysis without relying on a live download. Record the access date and whether the primary or archived data was used. The published list evolves over time.
# 
# **Dataset limitation:** This represents reported surgeries rather than all arm injuries (e.g., shoulder and flexor injuries are outside the scope). Earlier time periods may be less completely documented, and 2026 is an incomplete calendar year.

# ## 1. Exploratory Data Analysis (EDA)
# 
# We will inspect the raw data and its quality before cleaning. Initial type conversions for plotting are done **on a temporary copy** so we can detect data issues without changing the original rows.

# In[ ]:


from pathlib import Path
from io import StringIO
import csv
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from IPython.display import display

sns.set_theme(style="whitegrid", context="notebook")
pd.set_option("display.max_columns", 25)
print("pandas:", pd.__version__)


# In[ ]:


DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
DATA_PATH = DATA_DIR / "tj_surgeries.csv"
PRIMARY_URL = ("https://docs.google.com/spreadsheets/d/"
               "1gQujXQQGOVNaiuwSN680Hq-FDVsCwvN-3AazykOBON0/export?format=csv&gid=0")
BACKUP_URL = ("https://raw.githubusercontent.com/derrickvasquez/"
              "Tommy-John-Surgery-Analysis/Main/"
              "Updated%20Tommy%20John%20Surgery%20List%20%28%40MLBPlayerAnalys%29.csv")

def has_expected_header(csv_text):
    first_rows = list(csv.reader(StringIO(csv_text)))[:35]
    return any("Player" in row and "TJ Surgery Date" in row for row in first_rows)

if not DATA_PATH.exists():
    failures = []
    for label, url in [("Jon Roegele live spreadsheet", PRIMARY_URL),
                       ("archived GitHub snapshot", BACKUP_URL)]:
        try:
            response = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
            response.encoding = "utf-8-sig"
            if not has_expected_header(response.text):
                raise ValueError("Downloaded file did not contain the expected headers")
            DATA_PATH.write_text(response.text, encoding="utf-8")
            print(f"Downloaded {label} to {DATA_PATH}")
            break
        except (requests.RequestException, ValueError) as exc:
            failures.append(f"{label}: {exc}")
    else:
        raise RuntimeError(
            "Automatic download failed. Open the primary spreadsheet's 'TJ List' "
            "tab, select File > Download > Comma Separated Values (.csv), "
            "save it as data/tj_surgeries.csv, then rerun. Details: " + "; ".join(failures)
        )
else:
    print(f"Using existing CSV: {DATA_PATH} (commit this file for reviewers)")


# In[ ]:


def load_raw_surgery_list(path):
    # csv.reader tolerates the sheet's introductory donation row, which may
    # contain fewer columns than the real header.
    with open(path, encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    header_index = next((i for i, row in enumerate(rows[:35])
                         if "Player" in row and "TJ Surgery Date" in row), None)
    if header_index is None:
        raise ValueError("Could not find columns 'Player' and 'TJ Surgery Date'")
    headers = [col.strip() for col in rows[header_index]]
    n = len(headers)
    records = [(row[:n] + [""] * max(0, n - len(row))) for row in rows[header_index + 1:]
               if any(value.strip() for value in row)]
    df = pd.DataFrame(records, columns=headers)
    df = df.loc[:, ~pd.Index(df.columns).duplicated()].copy()
    df = df.loc[:, [bool(col.strip()) for col in df.columns]].copy()
    df = df.replace(r"^\s*$", np.nan, regex=True)
    df = df.loc[df["Player"].notna()].copy()
    df = df.loc[~df["Player"].astype(str).str.contains(
        "THE TOMMY JOHN SURGERY LIST HAS MOVED", case=False, na=False)].copy()
    df = df.reset_index(drop=True)
    return df

raw = load_raw_surgery_list(DATA_PATH)
print(f"Rows: {len(raw):,}; columns: {len(raw.columns)}")
display(raw.head())
print("Original dtypes:")
display(raw.dtypes.rename("dtype").to_frame())


# In[ ]:


print("Raw missing values (top 15):")
missing = raw.isna().sum().sort_values(ascending=False)
missing_pct = raw.isna().mean().mul(100).round(1)
display(pd.DataFrame({"Missing": missing, "Missing %": missing_pct}).head(15))
print("Entirely duplicated rows:", int(raw.duplicated().sum()))

# Normalize names for analysis; this is NOT the cleaning step yet.
column_mapping = {
    "Player": "player", "TJ Surgery Date": "surgery_date", "Team": "team",
    "Level": "level", "Majors": "level", "Position": "position",
    "Throws": "throws", "Age": "age", "Return Date (same level)": "return_date",
    "Recovery Time (months)": "recovery_months", "Country": "country",
    "mlbamid": "mlbamid", "Surgeon(s)": "surgeons"
}
eda = raw.rename(columns=column_mapping).copy()
if eda.columns.duplicated().any():
    eda = eda.loc[:, ~eda.columns.duplicated()].copy()
for field in ["player", "surgery_date", "level", "position", "throws", "age",
              "recovery_months", "return_date", "mlbamid"]:
    if field not in eda:
        eda[field] = pd.NA

eda["surgery_date"] = pd.to_datetime(eda["surgery_date"], errors="coerce")
eda["return_date"] = pd.to_datetime(eda["return_date"], errors="coerce")
eda["age"] = pd.to_numeric(eda["age"], errors="coerce")
eda["recovery_months"] = pd.to_numeric(eda["recovery_months"], errors="coerce")
eda["surgery_year"] = eda["surgery_date"].dt.year
eda["player_key"] = eda["player"].astype("string").str.strip().str.casefold()
print("Same player + surgery date repeated:", int(eda.duplicated(["player_key", "surgery_date"]).sum()))
print("Surgery dates that could not be parsed:", int(eda["surgery_date"].isna().sum()))
print("Ages outside plausible 12–60 range:", int(((eda["age"] < 12) | (eda["age"] > 60)).sum()))
print("Nonpositive recovery times:", int((eda["recovery_months"] <= 0).sum()))
print("Recovery times over 60 months (review, not automatically an error):",
      int((eda["recovery_months"] > 60).sum()))


# ### EDA: Distributions, relationships, and potential issues
# 
# The raw inspections above identify fields with missing information, possible duplicate surgery events, and values that require numeric/date conversion. A surgery event is identified by **player and date**, not player alone: one player may undergo multiple procedures. Plots below describe **reported case counts**, which are not injury rates.
# 
# **Graph 1 — Surgery counts over time:** Shows recorded surgery events by calendar year (2000–2025). We intentionally exclude 2026 because it is incomplete, and should not compare a partial year with full years.

# In[ ]:


annual_counts = (eda.loc[eda["surgery_year"].between(2000, 2025), "surgery_year"]
                   .value_counts().sort_index())
fig, ax = plt.subplots(figsize=(11, 4.5))
if len(annual_counts):
    sns.lineplot(x=annual_counts.index.astype(int), y=annual_counts.values,
                 marker="o", ax=ax)
    ax.set_xticks(list(range(2000, 2026, 2)))
    peak_year = int(annual_counts.idxmax())
    print(f"Highest observed annual count (2000–2025): {peak_year} ({annual_counts.max()} cases)")
else:
    ax.text(.5, .5, "No cases in 2000–2025", ha="center", va="center", transform=ax.transAxes)
ax.set(title="Documented Tommy John surgeries by year (2000–2025)",
       xlabel="Surgery year", ylabel="Number of recorded surgery events")
plt.tight_layout()
plt.show()


# **Interpretation:** This graph reveals changes in the *number of recorded cases*. In the final write-up, identify the peak year from the output. The rise or fall cannot automatically be interpreted as a change in individual injury risk because baseball population size and reporting completeness may also change.

# **Graph 2 — Age distribution:** A histogram of ages at the time of surgery. Ages outside 12–60 are hidden from the *plot* for readability and separately flagged for investigation; they are not silently removed from the raw dataset.

# In[ ]:


age_plot = eda["age"].dropna().loc[lambda s: s.between(12, 60)]
fig, ax = plt.subplots(figsize=(9, 4.5))
if not age_plot.empty:
    sns.histplot(age_plot, bins=range(12, 62, 2), color="steelblue", ax=ax)
    print(f"Median documented age at surgery: {age_plot.median():.1f} years; n={len(age_plot):,}")
else:
    ax.text(.5, .5, "No valid numeric ages", ha="center", transform=ax.transAxes)
ax.set(title="Age of players at Tommy John surgery", xlabel="Age (years)", ylabel="Recorded cases")
plt.tight_layout()
plt.show()


# **Interpretation:** The histogram shows which age ranges account for the largest *counts* of surgeries in the dataset. It does not establish age-specific injury risk without counts of all players of each age.

# **Graph 3 — Playing level:** Compares recorded cases across levels of competition, such as MLB and minor-league levels. Differences may partly reflect how many athletes participate at each level.

# In[ ]:


level_counts = (eda["level"].astype("string").str.strip().str.upper()
                .replace({"": pd.NA}).dropna().value_counts().head(12))
fig, ax = plt.subplots(figsize=(10, 5))
if len(level_counts):
    sns.barplot(x=level_counts.values, y=level_counts.index, ax=ax, color="cadetblue")
    print("Most frequently reported playing level:", level_counts.index[0],
          f"({int(level_counts.iloc[0])} cases)")
else:
    ax.text(.5, .5, "No documented levels", ha="center", transform=ax.transAxes)
ax.set(title="Recorded Tommy John surgeries by level", xlabel="Recorded surgery events", ylabel="Level")
plt.tight_layout()
plt.show()


# **Interpretation:** This plot describes the dataset's composition across levels, not how dangerous each level is. Different levels have different population sizes and reporting coverage.

# **Graph 4 — Recovery time vs. age:** A scatterplot shows the relationship between a player's recorded age and the reported months needed to return to the **same level**, among players for whom both measurements are known. Missing recovery values are excluded only from this plot; they do not mean zero recovery or failure to return.

# In[ ]:


recovery_plot = eda.loc[
    eda["age"].between(12, 60) & eda["recovery_months"].gt(0),
    ["age", "recovery_months"]
].dropna().copy()
fig, ax = plt.subplots(figsize=(9, 5))
if not recovery_plot.empty:
    sns.scatterplot(data=recovery_plot, x="age", y="recovery_months", alpha=.45, ax=ax)
    print(f"Cases with known age and recovery: {len(recovery_plot):,}")
    if len(recovery_plot) >= 3 and recovery_plot["age"].nunique() > 1 and recovery_plot["recovery_months"].nunique() > 1:
        print("Spearman correlation (age vs recovery time):",
              round(recovery_plot["age"].corr(recovery_plot["recovery_months"], method="spearman"), 3))
else:
    ax.text(.5, .5, "Recovery values are unavailable", ha="center", transform=ax.transAxes)
ax.set(title="Age vs. documented time to same-level return",
       xlabel="Age at surgery (years)", ylabel="Recovery time (months)")
plt.tight_layout()
plt.show()


# **Interpretation:** This plot tests for a possible *association* between age and recorded time to return. Even if a correlation appears, it does not show that age causes the difference: playing level, severity, rehab protocols, and selection of players with known return dates may contribute. Surgeries from recent years have not had equal follow-up time.

# **Graph 5 — Throwing hand:** Compares how many recorded surgery cases involve left-handed and right-handed throwers. Percentages are of *surgery cases*, not of the full baseball population.

# In[ ]:


hand = eda["throws"].astype("string").str.strip().str.upper()
hand_counts = hand[hand.isin(["R", "L"])].value_counts().reindex(["R", "L"], fill_value=0)
fig, ax = plt.subplots(figsize=(6.5, 4.5))
if hand_counts.sum() > 0:
    sns.barplot(x=["Right-handed", "Left-handed"], y=hand_counts.values,
                ax=ax, color="slateblue")
    print("Among recorded R/L surgery cases:",
          f"R={hand_counts['R']/hand_counts.sum():.1%}, L={hand_counts['L']/hand_counts.sum():.1%}")
else:
    ax.text(.5, .5, "No throwing-hand entries", ha="center", transform=ax.transAxes)
ax.set(title="Recorded Tommy John surgeries by throwing hand",
       xlabel="Throws", ylabel="Recorded surgery events")
plt.tight_layout()
plt.show()


# **Interpretation:** The difference in counts may reflect the much larger population of right-handed throwers. **Do not conclude** one handedness has higher risk unless we add the denominator (all eligible right- and left-handed players).

# ### Data-quality summary before cleaning
# 
# The checks below help answer the assignment's EDA questions about missing values, outliers, duplicates, and variable types. In particular, missing return/recovery times require special treatment. A player may not have returned yet, may never return to the same level, or may simply lack an updated report; these are **not interchangeable outcomes**.

# In[ ]:


quality_cols = ["player", "surgery_date", "age", "position", "level", "throws",
                "return_date", "recovery_months"]
summary = pd.DataFrame({
    "nonmissing": eda[quality_cols].notna().sum(),
    "missing": eda[quality_cols].isna().sum(),
    "missing_pct": (100 * eda[quality_cols].isna().mean()).round(1),
    "data_type_after_parsing": eda[quality_cols].dtypes.astype(str)
})
display(summary.sort_values("missing_pct", ascending=False))

print("Age quartiles / extremes:")
display(eda["age"].describe())
print("Recorded recovery-time quartiles / extremes:")
display(eda["recovery_months"].describe())
print("Possible problematic rows for manual review:")
review_mask = (~eda["age"].isna() & ~eda["age"].between(12, 60)) |               (eda["recovery_months"].notna() & ~eda["recovery_months"].between(0.01, 60))
display(eda.loc[review_mask, ["player", "surgery_date", "age", "recovery_months"]].head(12))


# ## 2. Data Cleaning and Transformation
# 
# **Changes motivated by EDA:**
# 
# 1. Standardize column names and text categories (e.g., uppercase `L` / `R` and playing levels).
# 2. Convert surgery/return dates into proper `datetime` fields, and age/recovery months into numeric values; invalid conversions become missing rather than made-up values.
# 3. Drop rows without a valid player or surgery date, which are unusable for case-level analysis over time.
# 4. Remove exact duplicate surgery events based on player ID (when available) or normalized name **plus surgery date**. Do **not** drop all repeated player names: repeat surgeries are real possibilities.
# 5. Mark impossible numeric values as missing, while **keeping plausible but unusually long recovery times** for review. Excluding genuine outliers could bias the results.
# 6. Leave missing return/recovery outcomes missing for now. Fill a missing number of months **only when a valid same-level return date is supplied**, computing the elapsed months directly from the two dates.
# 7. Add `surgery_year` and a flag describing where recovery information came from. The cleaned CSV is saved for reproducibility.

# In[ ]:


clean = eda.copy()
starting_rows = len(clean)
clean["player"] = clean["player"].astype("string").str.strip()
for col in ["level", "position", "throws"]:
    clean[col] = clean[col].astype("string").str.strip().str.upper()
    clean[col] = clean[col].replace({"": pd.NA, "NAN": pd.NA, "NONE": pd.NA})

# Keep only cases with an identifiable player and valid surgery date.
clean = clean.loc[clean["player"].notna() & clean["surgery_date"].notna()].copy()
removed_unusable = starting_rows - len(clean)

# These ages are implausible for organized baseball and more likely entry errors.
invalid_age = clean["age"].notna() & ~clean["age"].between(12, 60)
clean.loc[invalid_age, "age"] = np.nan
invalid_recovery = clean["recovery_months"].notna() & (clean["recovery_months"] <= 0)
clean.loc[invalid_recovery, "recovery_months"] = np.nan

# Calculate recovery months from a documented return date only where missing.
valid_elapsed_days = (clean["return_date"] - clean["surgery_date"]).dt.days
can_derive = (clean["recovery_months"].isna() & valid_elapsed_days.gt(0))
clean["recovery_source"] = np.where(clean["recovery_months"].notna(), "reported", "unknown")
clean.loc[can_derive, "recovery_months"] = valid_elapsed_days[can_derive] / 30.4375
clean.loc[can_derive, "recovery_source"] = "derived_from_dates"

# Prefer unique MLBAM ID where present, otherwise use a normalized name.
clean["player_key"] = clean["player"].str.casefold().str.replace(r"[^a-z0-9]", "", regex=True)
id_values = clean["mlbamid"].astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA})
clean["case_player_key"] = ("id:" + id_values).fillna("name:" + clean["player_key"])
clean = clean.sort_values(["case_player_key", "surgery_date", "recovery_months"],
                          na_position="last")
pre_dedup_rows = len(clean)
clean = clean.drop_duplicates(subset=["case_player_key", "surgery_date"], keep="first").copy()
removed_duplicate_events = pre_dedup_rows - len(clean)

clean["surgery_year"] = clean["surgery_date"].dt.year.astype("Int64")
clean["age"] = clean["age"].astype("Float64")
clean["recovery_months"] = clean["recovery_months"].astype("Float64")
clean["unusually_long_recovery"] = clean["recovery_months"].gt(60).fillna(False)

print(f"Starting records: {starting_rows:,}")
print(f"Dropped records without identifiable player/date: {removed_unusable:,}")
print(f"Removed duplicate surgery events: {removed_duplicate_events:,}")
print(f"Ages converted to missing after anomaly check: {int(invalid_age.sum()):,}")
print(f"Nonpositive recovery values changed to missing: {int(invalid_recovery.sum()):,}")
print(f"Recovery times derived from known return dates: {int(can_derive.sum()):,}")
print(f"Long recovery times (>60 months) flagged for review: {int(clean['unusually_long_recovery'].sum()):,}")
print(f"Cleaned records: {len(clean):,}")

display(clean[["player", "surgery_date", "age", "level", "throws",
               "recovery_months", "recovery_source", "surgery_year"]].head(10))
OUTPUT_CSV = DATA_DIR / "tj_surgeries_cleaned.csv"
clean.to_csv(OUTPUT_CSV, index=False)
print("Saved:", OUTPUT_CSV)


# ### Post-cleaning checks and updated observations
# 
# Print exact descriptive findings from the cleaned records. These numbers update if the source data is refreshed. Avoid making causal or medical claims from case-only data.

# In[ ]:


full_years = clean.loc[clean["surgery_year"].between(2000, 2025)]
annual_clean = full_years.groupby("surgery_year").size().sort_index()
if not annual_clean.empty:
    print(f"2000–2025: {len(full_years):,} documented surgery events")
    print(f"Peak recorded year: {int(annual_clean.idxmax())}, "
          f"{int(annual_clean.max())} events")
print("2026 events currently recorded (partial year):",
      int(clean["surgery_year"].eq(2026).sum()))
print("Age median:", clean["age"].median())
print("Share with reported/derived same-level recovery times:",
      f"{clean['recovery_months'].notna().mean():.1%}")
known_returns = clean.loc[clean["recovery_months"].notna()]
if not known_returns.empty:
    print(f"Median documented same-level recovery: "
          f"{known_returns['recovery_months'].median():.1f} months "
          f"(n={len(known_returns):,}; selected subset)")

print("Additional missing-value rate after cleaning:")
display((clean[["age", "level", "throws", "recovery_months"]].isna()
         .mean().mul(100).round(1).rename("Missing %").to_frame()))


# ## 3. Machine Learning Plan
# 
# ### Proposed model and prediction target
# 
# I plan to start with a **supervised regression model** predicting the **number of months to a documented same-level return** after Tommy John surgery. Input features would include age at surgery, throwing hand, position, competition level, and surgery year. These are variables known on or before the surgery date; I will **not** use return dates or any information from after surgery as predictors.
# 
# - **Baseline:** `DummyRegressor(strategy="median")` to measure how well a simple constant prediction performs.
# - **Initial model:** linear regression or regularized Ridge regression with one-hot encoded categorical variables.
# - **Comparison:** random forest regression to detect possible nonlinear relationships.
# - **Evaluation:** mean absolute error (MAE) in months, alongside the baseline. Group by player when splitting repeated surgeries, and consider an out-of-time test (older surgery years for training, more recent completed follow-up years for testing).
# 
# ### Anticipated challenges and solutions
# 
# **Missing recovery outcomes / selection bias:** A blank return date does **not** imply that a player failed to return. Training only on recorded returns means predictions apply to *documented returners*, not all surgery patients. I will explicitly state this limitation and examine missingness by surgery year and level. The more recent surgeries must be excluded from time-to-return training because they have had less follow-up time.
# 
# **Small or unbalanced subsets:** Certain levels and non-pitcher positions may have few cases. I will combine rare categories if necessary, evaluate error by subgroup, and avoid overconfident conclusions.
# 
# **Duplicates / repeat surgery and leakage:** I will deduplicate records for the same surgery but preserve separate surgeries. I will split data in a way that avoids putting the same player's multiple records into both training and test sets.
# 
# **Limited prediction strength:** This data does not include mechanics, workload, velocity, injury severity, medical treatment, or rehabilitation details. Thus even a working model might perform poorly. I will compare against the median baseline and report the limitations rather than claiming the model explains why injuries happen.
# 
# **Missing predictor values / outliers:** Numeric features will be imputed using **training-only** medians; categorical values will receive an `Unknown` category during pipeline preprocessing. Plausible extreme recovery times will be checked and model sensitivity to them will be reported.
# 
# **Alternative if the regression data are too incomplete:** Keep the project descriptive or model a more restricted target, such as **documented recovery time ≥18 months among players who returned to the same level**, with appropriate class balance and selection-bias warnings. I will *not* treat nonreturners or missing follow-up as negative class examples.

# In[ ]:


# Feasibility check ONLY; actual model training will be in the final project.
# An older cohort reduces, but does not eliminate, follow-up bias.
model_candidate = clean.loc[
    clean["surgery_year"].between(2000, 2022) &
    clean["recovery_months"].notna() &
    clean["recovery_months"].gt(0)
].copy()
features = ["age", "throws", "position", "level", "surgery_year"]
print("Candidate records for same-level return-time regression:", len(model_candidate))
print("Candidate target median, in months:", model_candidate["recovery_months"].median())
print("Predictor missingness in candidate pool:")
display(model_candidate[features].isna().mean().mul(100).round(1))
print("Number of separate players in candidate pool:", model_candidate["case_player_key"].nunique())
print("Note: this cohort excludes players without documented recovery outcomes; it is selected.")


# ## 4. Prior Feedback and Updates
# 
# > **Complete this with your real Checkpoint #1 comments. Do not fabricate peer or instructor feedback.**
# 
# **Feedback received:** [Insert the actual feedback from peers or teaching team. If none has been received, write: "No feedback had been received at the time of this checkpoint."]
# 
# **Changes made based on feedback:** [State the corresponding edits you actually made. If none, write: "No feedback-driven changes yet."]
# 
# **Independent updates since Checkpoint #1:** I narrowed the broad topic of baseball arm injuries to documented Tommy John surgery cases so that the research question and measurable outcome variables are clearer. I added a public, case-level dataset; performed exploratory analysis and multiple visualizations; checked missing data, duplicate surgery cases and outliers; and created a machine-learning plan focused on recovery time. I will continue refining this based on comments and data quality.
# 
# **Remaining work for the final project:** Finish validating the cleaned data, assess whether known recovery outcomes are representative, test baseline and regression models, compare performance and discuss limitations.
# 
# ### Reflection (fill in after running all cells)
# 
# - Strongest finding: [use an actual count/median/trend from the charts above].
# - Important data-quality issue: [state the actual missingness percentage shown above].
# - Next change: [describe your most useful improvement for the final project].

# ## Submission checklist
# 
# - [ ] Preserve Checkpoint #1 sections, add these Checkpoint #2 cells, and incorporate real feedback.
# - [ ] Run **all** cells; check that all **five graphs** display with actual data and interpret outputs.
# - [ ] Add specific observations to your write-up rather than leaving placeholders.
# - [ ] Commit `source.ipynb`, `source.py`, `data/tj_surgeries.csv`, and optionally `data/tj_surgeries_cleaned.csv` to your **existing GitHub repository**.
# - [ ] Keep the following conversion command as the **last cell** of your final `source.ipynb`. Re-run it after editing the notebook, then commit the updated script.
# - [ ] Submit the **same GitHub repository URL** as Checkpoint #1.
# 
# **Due:** October 25, 2026. **Peer review due:** November 1, 2026.

# In[ ]:


get_ipython().system('jupyter nbconvert --to python source.ipynb')

