import pandas as pd
import numpy as np
from statsmodels.multivariate.manova import MANOVA
from scipy.stats import shapiro, levene
from const import scales
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Load processed data
df = pd.read_csv('data/processed_data.csv', sep=',', encoding='utf-8')
group_col = 'robot_exp_group'

# Define prior social robot experience group
def exp_group(val):
    if val in [1, 2]:
        return 'None'
    elif val in [3, 4, 5, 6]:
        return 'Any'
    else:
        return np.nan

df[group_col] = df['RS01_01'].apply(exp_group)
print(df[group_col].value_counts(dropna=False))


# Calculate subscale means
for scale, items in scales.items():
    valid_items = [col for col in items if col in df.columns]
    if not valid_items:
        print(f"Warning: No valid items found for scale '{scale}'")
        continue
    df[scale] = df[valid_items].mean(axis=1)

subscale_names = list(scales.keys())
groups = df[group_col].dropna().unique()


### Export descriptive statistics (mean, std) for each group and subscale, and count once at the beginning
desc_stats = df.groupby(group_col)[subscale_names].agg(['mean', 'std'])
counts = df.groupby(group_col).size().rename('count')
desc_stats_with_count = pd.concat([counts, desc_stats], axis=1)
desc_stats_with_count.to_csv('results/rq2_group_descriptives.csv')


### Assumption checks
print("\nShapiro-Wilk normality test for each subscale (by group):")
for scale in subscale_names:
    for group in groups:
        vals = df[df[group_col] == group][scale].dropna()
        if len(vals) > 3:
            stat, p = shapiro(vals)
            print(f"{scale} ({group}): W={stat:.3f}, p={p:.3f}")

print("\nLevene's test for homogeneity of variances:")
for scale in subscale_names:
    vals = [df[df[group_col] == group][scale].dropna() for group in groups]
    if all(len(v) > 3 for v in vals):
        stat, p = levene(*vals)
        print(f"{scale}: W={stat:.3f}, p={p:.3f}")


### Run MANOVA
dv_formula = ' + '.join(subscale_names)
formula = f'{dv_formula} ~ {group_col}'
print("\nMANOVA results:")
maov = MANOVA.from_formula(formula, data=df)
manova_results = maov.mv_test()


### Export Wilks' Lambda tables for intercept and robot_exp_group
def extract_multivariate_tests(manova_results, effect_names):
    rows = []
    for effect in effect_names:
        stat_table = manova_results.results[effect]['stat']
        for test_name, row in stat_table.iterrows():
            row_out = row.copy()
            row_out['Effect'] = effect
            row_out['Test'] = test_name
            rows.append(row_out)
    return pd.DataFrame(rows)

effect_names = ['Intercept', group_col]
multivariate_tests_df = extract_multivariate_tests(manova_results, effect_names)
multivariate_tests_df.to_csv('results/rq2_manova_multivariate_tests.csv', index=False)

print(multivariate_tests_df)



# Post-hoc ANOVAs and Tukey HSD
# print("\nPost-hoc ANOVAs and Tukey HSD (if significant):")
# for scale in subscale_names:
#     model = ols(f'{scale} ~ C({group_col})', data=df).fit()
#     anova_table = sm.stats.anova_lm(model, typ=2)
#     print(f"\n{scale} ANOVA:")
#     print(anova_table)
#     if anova_table['PR(>F)'][0] < 0.05:
#         print(f"Tukey HSD for {scale}:")
#         tukey = pairwise_tukeyhsd(df[scale], df[group_col])
#         print(tukey)



### Visualization
df_melt = df.melt(id_vars=[group_col], value_vars=subscale_names, var_name='Subscale', value_name='Score')
plt.figure(figsize=(10, 6))
sns.barplot(data=df_melt, x='Subscale', y='Score', hue=group_col, ci='sd')
plt.title('Attitude Subscales by Social Robot Experience')
plt.ylabel('Mean Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('plots/attitude_subscales_by_experience.png')