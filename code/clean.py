"""Data cleaning: transform the raw survey export into the analytical dataset.

Reads data/raw_data.xlsx (a two-header-row wide export from the survey
platform) and produces data/cleaned_data.csv, a tidy long-format dataset
with one row per participant and one column per variable.

The raw export is a "wide" file: each of the six experimental conditions
(stakes x explanation) has its own block of item columns, and each
participant only answered the block matching their randomly assigned
condition. This script maps each participant to their condition block,
extracts the answered items, and appends demographics and controls.
"""

import pandas as pd
import numpy as np

RAW = "data/raw_data.xlsx"
OUT = "data/cleaned_data.csv"

# ------------------------------------------------------------------ load
raw = pd.read_excel(RAW, header=None)
header_l2 = raw.iloc[1].tolist()          # second header row = item labels
data = raw.iloc[2:].reset_index(drop=True)

# ------------------------------------------------- condition from "随机元素"
def parse_condition(s):
    """Normalize the inconsistent '随机元素' strings into (stakes, explanation)."""
    t = str(s).replace(" ", "")
    stakes = "High Stakes" if "High" in t else "Low Stakes"
    if "Empathic" in t:
        expl = "Empathic"
    elif "Process" in t:
        expl = "Process"
    else:
        expl = "Outcome"
    return stakes, expl

cond = data[17].apply(parse_condition)

# ------------------------------------------------------ condition -> columns
# each block holds Q(s)/PIL1-4/IT1-3/PA1-3 with per-item time columns
block_cols = {
    ("Low Stakes", "Outcome"):   range(18, 42),
    ("Low Stakes", "Process"):   range(42, 64),
    ("Low Stakes", "Empathic"):  range(64, 86),
    ("High Stakes", "Outcome"):  range(86, 108),
    ("High Stakes", "Process"):  range(108, 130),
    ("High Stakes", "Empathic"): range(130, 152),
}
items = ["PIL1", "PIL2", "PIL3", "PIL4", "IT1", "IT2", "IT3",
         "PA1", "PA2", "PA3"]

def block_item_map(cols):
    """Return {item_label: column_index} for a condition block."""
    m = {}
    for c in cols:
        label = str(header_l2[c]).strip()
        if label in items:
            m[label] = c
    return m

item_maps = {k: block_item_map(v) for k, v in block_cols.items()}

# ------------------------------------------------------- demographics/control
# fixed columns (second header): Q1=gender, Q2=age, Q3=edu, Q4=job,
# GT1-3, AA1-2, AE1-2, PAC1-2 (labeled PA1/PA2 in the raw file)
gender_col, age_col, edu_col, job_col = 152, 154, 156, 158
gt_cols = [160, 162, 164]
aa_cols = [166, 168]
ae_cols = [170, 172]
pac_cols = [174, 176]

# --------------------------------------------------------------- build output
out = pd.DataFrame()
out["Stakes_Group"] = cond.str[0]
out["Explanation_Group"] = cond.str[1]

for it in items:
    out[it] = [
        pd.to_numeric(data.loc[i, item_maps[(out.loc[i, "Stakes_Group"],
                                             out.loc[i, "Explanation_Group"])][it]],
                      errors="coerce")
        for i in data.index
    ]

out["Gender"] = data[gender_col].values
out["Age_Group"] = data[age_col].values
out["Edu_Level"] = data[edu_col].values
out["Job_Type"] = data[job_col].values

for name, cols in [("GT", gt_cols), ("AA", aa_cols), ("AE", ae_cols), ("PAC", pac_cols)]:
    for j, c in enumerate(cols, start=1):
        out[f"{name}{j}"] = pd.to_numeric(data[c], errors="coerce")

# composite means
for prefix, n in [("PIL", 4), ("IT", 3), ("PA", 3), ("GT", 3), ("AA", 2), ("AE", 2)]:
    item_cols = [f"{prefix}{i}" for i in range(1, n + 1)]
    out[f"{prefix}_Mean"] = out[item_cols].mean(axis=1)

# reorder columns to match the canonical schema
col_order = (["Stakes_Group", "Explanation_Group"]
             + [f"PIL{i}" for i in range(1, 5)] + ["PIL_Mean"]
             + [f"IT{i}" for i in range(1, 4)] + ["IT_Mean"]
             + [f"PA{i}" for i in range(1, 4)] + ["PA_Mean"]
             + ["Gender", "Age_Group", "Edu_Level", "Job_Type"]
             + [f"GT{i}" for i in range(1, 4)] + ["GT_Mean"]
             + [f"AA{i}" for i in range(1, 3)] + ["AA_Mean"]
             + [f"AE{i}" for i in range(1, 3)] + ["AE_Mean"]
             + ["PAC1", "PAC2"])
out = out[col_order]

# ---------------------------------------------------------------- validate
assert out.shape[0] == 420, f"expected 420 rows, got {out.shape[0]}"
assert out["Stakes_Group"].isna().sum() == 0
assert out["Explanation_Group"].isna().sum() == 0
missing = out[[f"{p}{i}" for p, n in [("PIL", 4), ("IT", 3), ("PA", 3)]
               for i in range(1, n + 1)]].isna().sum().sum()
assert missing == 0, f"expected 0 missing item values, got {missing}"

out.to_csv(OUT, index=False)
print(f"Wrote {OUT}: {out.shape[0]} rows x {out.shape[1]} columns")
print("Group sizes:")
print(out.groupby(["Stakes_Group", "Explanation_Group"]).size())
