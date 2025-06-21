import pandas as pd
import numpy as np
from statsmodels.multivariate.manova import MANOVA
from scipy.stats import shapiro, levene
from const import scales
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Load processed data
df = pd.read_csv('data/processed_data.csv', sep=',', encoding='utf-8')

# Calculate subscale means
for scale, items in scales.items():
    valid_items = [col for col in items if col in df.columns]
    if not valid_items:
        print(f"Warning: No valid items found for scale '{scale}'")
        continue
    df[scale] = df[valid_items].mean(axis=1)

subscale_names = list(scales.keys())



### Export descriptive statistics
desc_stats = df.groupby(['early_training', 'robot_exp_group'])[subscale_names].agg(['mean', 'std'])
counts = df.groupby(['early_training', 'robot_exp_group']).size().rename('count')
desc_stats_with_count = pd.concat([counts, desc_stats], axis=1)
desc_stats_with_count.to_csv('results/rq3_group_descriptives.csv')


### MANOVA
dv_formula = ' + '.join(subscale_names)
formula = f'{dv_formula} ~ early_training * robot_exp_group'

print("\nTwo-Way MANOVA results:")
maov = MANOVA.from_formula(formula, data=df)
manova_results = maov.mv_test()

### Export Wilks' Lambda tables 
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

effect_names = [
    'Intercept',
    'early_training',
    'robot_exp_group',
    'early_training:robot_exp_group'
]

multivariate_tests_df = extract_multivariate_tests(manova_results, effect_names)
multivariate_tests_df.to_csv('results/rq3_manova_multivariate_tests.csv', index=False)

print(multivariate_tests_df)


# Optional: Post-hoc ANOVAs for each subscale if significant
# print("\nPost-hoc ANOVAs for each subscale:")
# for scale in subscale_names:
#     model = ols(f'{scale} ~ early_training * robot_exp_group', data=df).fit()
#     anova_table = sm.stats.anova_lm(model, typ=2)
#     print(f"\n{scale} ANOVA:")
#     print(anova_table)