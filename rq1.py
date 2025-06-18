import pandas as pd
import numpy as np
from statsmodels.multivariate.manova import MANOVA
from scipy.stats import shapiro, levene
from const import scales
import matplotlib.pyplot as plt
import seaborn as sns

# Load processed data
df = pd.read_csv('data/processed_data.csv', sep=',', encoding='utf-8')
group_col = 'early_training'

# Define early musical training variable (adjust if needed)
df[group_col] = df['SD05'].map({1: 'Yes', 2: 'No'}) 


# Calculate subscale means
for scale, items in scales.items():
    # Ensure items are in the DataFrame columns
    valid_items = [col for col in items if col in df.columns]
    if not valid_items:
        print(f"Warning: No valid items found for scale '{scale}'")
        continue
    # Calculate mean for the scale
    df[scale] = df[valid_items].mean(axis=1)

# Prepare data for MANOVA
subscale_names = list(scales.keys())
groups = df[group_col].dropna().unique()


### Export descriptive statistics (mean, std) for each group and subscale, and count once at the beginning
desc_stats = df.groupby(group_col)[subscale_names].agg(['mean', 'std'])

# Add count as a separate row at the top
counts = df.groupby(group_col).size().rename('count')
desc_stats_with_count = pd.concat([counts, desc_stats], axis=1)

desc_stats_with_count.to_csv('results/rq1_group_descriptives.csv')


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


### Prepare MANOVA formula
dv_formula = ' + '.join(subscale_names)
formula = f'{dv_formula} ~ early_training'

# Run MANOVA
print("\nMANOVA results:")
maov = MANOVA.from_formula(formula, data=df)
manova_results = maov.mv_test()


### Extract Wilks' Lambda tables for intercept and early_training
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

effect_names = ['Intercept', 'early_training']
multivariate_tests_df = extract_multivariate_tests(manova_results, effect_names)
multivariate_tests_df.to_csv('results/rq1_manova_multivariate_tests.csv', index=False)

print(multivariate_tests_df)


# # If significant, run post-hoc ANOVAs
# print("\nPost-hoc ANOVAs:")
# import statsmodels.api as sm
# from statsmodels.formula.api import ols

# for scale in subscale_names:
#     model = ols(f'{scale} ~ early_training', data=df).fit()
#     anova_table = sm.stats.anova_lm(model, typ=2)
#     print(f"\n{scale} ANOVA:")
#     print(anova_table)

# # Effect sizes (eta squared)
# def eta_squared(aov):
#     return aov['sum_sq']['early_training'] / (aov['sum_sq']['early_training'] + aov['sum_sq']['Residual'])

# for scale in subscale_names:
#     model = ols(f'{scale} ~ early_training', data=df).fit()
#     aov = sm.stats.anova_lm(model, typ=2)
#     print(f"{scale} eta squared: {eta_squared(aov):.3f}")



### Visualization
df_melt = df.melt(id_vars=[group_col], value_vars=subscale_names, var_name='Subscale', value_name='Score')
plt.figure(figsize=(10, 6))
sns.barplot(data=df_melt, x='Subscale', y='Score', hue=group_col, ci='sd')
plt.title('Attitude Subscales by Early Musical Training')
plt.ylabel('Mean Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('plots/attitude_subscales_by_training.png')