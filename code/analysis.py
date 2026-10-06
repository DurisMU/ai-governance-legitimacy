"""Analysis script for the AI-governance legitimacy experiment.

Reproducible from data/cleaned_data.csv.
Runs: reliability (Cronbach's alpha), descriptive statistics, 2x3 ANOVA,
effect sizes (partial eta-squared), simple effects, post-hoc tests,
and produces figures saved to figures/.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from itertools import combinations

# ---------------------------------------------------------------- paths
DATA = "data/cleaned_data.csv"
FIG_DIR = "figures"

# ------------------------------------------------------------ helpers
def cronbach_alpha(df, items):
    """Cronbach's alpha for a set of item columns."""
    X = df[items].dropna()
    k = X.shape[1]
    if k < 2:
        return np.nan
    var_sum = X.var(axis=0, ddof=1).sum()
    var_total = X.sum(axis=1).var(ddof=1)
    return (k / (k - 1)) * (1 - var_sum / var_total)


def partial_eta2(aov_table, effect, residual="Residual"):
    ss_effect = aov_table.loc[effect, "sum_sq"]
    ss_resid = aov_table.loc[residual, "sum_sq"]
    return ss_effect / (ss_effect + ss_resid)


def run_anova(df, dv):
    formula = f"{dv} ~ C(Stakes_Group) * C(Explanation_Group)"
    model = ols(formula, data=df).fit()
    aov = sm.stats.anova_lm(model, typ=2)
    out = {}
    for term in ["C(Stakes_Group)", "C(Explanation_Group)",
                 "C(Stakes_Group):C(Explanation_Group)"]:
        row = aov.loc[term]
        out[term] = {
            "F": row["F"],
            "df": int(row["df"]),
            "p": row["PR(>F)"],
            "eta2": partial_eta2(aov, term),
        }
    return model, aov, out


# -------------------------------------------------------------- load
df = pd.read_csv(DATA)

# recode categorical variables to English
gender_map = {"男": "Male", "女": "Female"}
age_map = {"0-20岁": "Under 20", "21-30岁": "21-30", "31-40岁": "31-40",
           "41-50岁": "41-50", "51-60岁": "51-60", "60岁以上": "60+"}
edu_map = {"初中": "Junior high", "普高/中专/技校/职高": "High school",
           "专科": "Associate", "本科": "Bachelor", "硕士": "Master",
           "博士": "Doctorate"}

df["Gender"] = df["Gender"].map(gender_map)
df["Age_Group"] = df["Age_Group"].map(age_map)
df["Edu_Level"] = df["Edu_Level"].map(edu_map)

DVs = {"PIL": "Perceived Institutional Legitimacy",
       "IT": "Institutional Trust",
       "PA": "Policy Acceptance"}

# ---------------------------------------------------- 1. reliability
print("=" * 70)
print("1. RELIABILITY (Cronbach's alpha)")
print("=" * 70)
scales = {
    "PIL (legitimacy)": ["PIL1", "PIL2", "PIL3", "PIL4"],
    "IT (trust)": ["IT1", "IT2", "IT3"],
    "PA (acceptance)": ["PA1", "PA2", "PA3"],
    "GT (gov trust)": ["GT1", "GT2", "GT3"],
    "AA (AI attitude)": ["AA1", "AA2"],
    "AE (AI experience)": ["AE1", "AE2"],
    "PAC (policy attention)": ["PAC1", "PAC2"],
}
reliability = {}
for name, items in scales.items():
    a = cronbach_alpha(df, items)
    reliability[name] = a
    print(f"  {name:22s}: alpha = {a:.3f}")
reliability_df = pd.DataFrame(
    {"Scale": list(reliability.keys()),
     "Items": [len(scales[k]) for k in reliability],
     "Cronbach_alpha": [round(v, 3) for v in reliability.values()]}
)

# ------------------------------------------- 2. descriptive stats
print()
print("=" * 70)
print("2. DESCRIPTIVE STATISTICS (mean and SD by group)")
print("=" * 70)
groups = df.groupby(["Stakes_Group", "Explanation_Group"])
agg_spec = {"n": ("PIL_Mean", "size")}
for dv in DVs:
    agg_spec[f"{dv}_mean"] = (f"{dv}_Mean", "mean")
    agg_spec[f"{dv}_sd"] = (f"{dv}_Mean", "std")
desc = groups.agg(**agg_spec)
desc = desc.reindex(
    pd.MultiIndex.from_product(
        [["Low Stakes", "High Stakes"],
         ["Outcome", "Process", "Empathic"]],
        names=["Stakes_Group", "Explanation_Group"]
    )
)
print(desc.round(3).to_string())
desc.to_csv("data/descriptive_stats.csv")

# ------------------------------------------- 3. 2x3 ANOVA
print()
print("=" * 70)
print("3. 2 x 3 ANOVA (per dependent variable)")
print("=" * 70)
anova_rows = []
for key, label in DVs.items():
    dv = f"{key}_Mean"
    model, aov, res = run_anova(df, dv)
    print(f"\n--- {label} ({dv}) ---")
    print(aov.round(4).to_string())
    for term, r in res.items():
        stars = "***" if r["p"] < 0.001 else "**" if r["p"] < 0.01 else "*" if r["p"] < 0.05 else "n.s."
        print(f"  {term:40s} F({int(r['df'])},{int(aov.loc['Residual','df'])}) "
              f"= {r['F']:.3f}, p = {r['p']:.4g} {stars}, partial eta2 = {r['eta2']:.3f}")
        anova_rows.append({
            "DV": label, "Effect": term, "F": round(r["F"], 3),
            "df_num": int(r["df"]), "df_den": int(aov.loc["Residual", "df"]),
            "p": r["p"], "partial_eta2": round(r["eta2"], 3),
        })

anova_df = pd.DataFrame(anova_rows)
anova_df.to_csv("data/anova_results.csv", index=False)

# ------------------------------------------- 4. simple effects
print()
print("=" * 70)
print("4. SIMPLE EFFECTS")
print("=" * 70)
# simple effect of Explanation within each Stakes level
for stakes in ["Low Stakes", "High Stakes"]:
    sub = df[df["Stakes_Group"] == stakes]
    print(f"\n  [Explanation simple effect within {stakes}]")
    for key, label in DVs.items():
        dv = f"{key}_Mean"
        m = ols(f"{dv} ~ C(Explanation_Group)", data=sub).fit()
        a = sm.stats.anova_lm(m, typ=2)
        f = a.loc["C(Explanation_Group)", "F"]
        p = a.loc["C(Explanation_Group)", "PR(>F)"]
        print(f"    {key}: F(2,{int(a.loc['Residual','df'])}) = {f:.3f}, p = {p:.4g}")

# simple effect of Stakes within each Explanation level
for expl in ["Outcome", "Process", "Empathic"]:
    sub = df[df["Explanation_Group"] == expl]
    print(f"\n  [Stakes simple effect within {expl}]")
    for key, label in DVs.items():
        dv = f"{key}_Mean"
        m = ols(f"{dv} ~ C(Stakes_Group)", data=sub).fit()
        a = sm.stats.anova_lm(m, typ=2)
        f = a.loc["C(Stakes_Group)", "F"]
        p = a.loc["C(Stakes_Group)", "PR(>F)"]
        print(f"    {key}: F(1,{int(a.loc['Residual','df'])}) = {f:.3f}, p = {p:.4g}")

# ------------------------------------------- 5. post-hoc (Tukey)
print()
print("=" * 70)
print("5. POST-HOC (Tukey HSD on Explanation, per Stakes)")
print("=" * 70)
for stakes in ["Low Stakes", "High Stakes"]:
    sub = df[df["Stakes_Group"] == stakes].copy()
    sub["cell"] = sub["Stakes_Group"] + " / " + sub["Explanation_Group"]
    print(f"\n  {stakes}:")
    for key in DVs:
        dv = f"{key}_Mean"
        th = pairwise_tukeyhsd(sub[dv], sub["Explanation_Group"])
        print(f"    {key}:")
        for r in th.summary().data[1:]:
            g1, g2, meandiff, p, lower, upper, reject = r
            print(f"      {g1} vs {g2}: diff={meandiff:.3f}, p={p:.4g}")

# ------------------------------------------- 6. control variables
print()
print("=" * 70)
print("6. CONTROL VARIABLE FREQUENCIES")
print("=" * 70)
for col in ["Gender", "Age_Group", "Edu_Level"]:
    print(f"\n  {col}:")
    print(df[col].value_counts().to_string())

# ------------------------------------------------- 7. figures
print()
print("=" * 70)
print("7. FIGURES")
print("=" * 70)
sns.set_theme(style="whitegrid", context="talk")

# --- interaction plot (3 DVs in one row)
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
expl_order = ["Outcome", "Process", "Empathic"]
for ax, (key, label) in zip(axes, DVs.items()):
    dv = f"{key}_Mean"
    sns.pointplot(data=df, x="Explanation_Group", y=dv,
                  hue="Stakes_Group", order=expl_order,
                  hue_order=["Low Stakes", "High Stakes"],
                  errorbar="se", capsize=0.1, ax=ax, markers=["o", "s"],
                  legend=False)
    ax.set_title(label)
    ax.set_xlabel("Explanation type")
    ax.set_ylim(1, 5.2)
axes[0].set_ylabel("Mean rating")
handles, _ = axes[0].get_legend_handles_labels()
fig.legend(handles, ["Low stakes", "High stakes"], title="Decision stakes",
           loc="lower center", ncol=2, frameon=False,
           bbox_to_anchor=(0.5, -0.05))
plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig(f"{FIG_DIR}/interaction_plot.png", dpi=200, bbox_inches="tight")
plt.close()
print("  saved figures/interaction_plot.png")

# --- boxplot (legitimacy)
fig, ax = plt.subplots(figsize=(8, 4.5))
sns.boxplot(data=df, x="Explanation_Group", y="PIL_Mean", hue="Stakes_Group",
            order=expl_order, hue_order=["Low Stakes", "High Stakes"], ax=ax)
ax.set_title("Perceived Institutional Legitimacy by condition")
ax.set_xlabel("Explanation type")
ax.set_ylabel("PIL mean")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/boxplot_pil.png", dpi=200, bbox_inches="tight")
plt.close()
print("  saved figures/boxplot_pil.png")

# --- correlation heatmap (DV means + controls)
corr_cols = [f"{k}_Mean" for k in DVs] + ["GT_Mean", "AA_Mean", "AE_Mean"]
corr = df[corr_cols].corr()
labels = ["Legitimacy", "Trust", "Acceptance",
          "Gov. Trust", "AI Attitude", "AI Experience"]
corr.index = labels
corr.columns = labels
fig, ax = plt.subplots(figsize=(6.5, 5.5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0,
            square=True, ax=ax, cbar_kws={"shrink": 0.8})
ax.set_title("Correlations among key measures")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/correlation_heatmap.png", dpi=200, bbox_inches="tight")
plt.close()
print("  saved figures/correlation_heatmap.png")

print()
print("Done. All results and figures written.")
