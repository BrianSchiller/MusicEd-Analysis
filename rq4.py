import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from const import scales
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
from statsmodels.stats.power import GofChisquarePower
import numpy as np


# Load processed data
df = pd.read_csv('data/processed_data.csv', sep=',', encoding='utf-8')

# Calculate subscale means (if not already present)
for scale, items in scales.items():
    valid_items = [col for col in items if col in df.columns]
    if not valid_items:
        continue
    df[scale] = df[valid_items].mean(axis=1)

subscale_names = list(scales.keys())

# Prepare data for clustering
X = df[subscale_names].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-means clustering
k = 5
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df.loc[X.index, 'cluster'] = kmeans.fit_predict(X_scaled)

print("\nCluster sizes:")
print(df['cluster'].value_counts())

### Export cluster profiles (mean subscale scores) as CSV
cluster_profiles = df.groupby('cluster')[subscale_names].mean()
cluster_profiles.to_csv('results/cluster_profiles_mean_subscales.csv')

print("\nCluster profiles (mean subscale scores):")
print(cluster_profiles)



### Profile clusters by demographics (adjust column names as needed)
demographic_map = {
    'SD01_08': 'age',
    'SD02': 'gender',
    'SD04': 'education',
    'early_training': 'musical_education',
    'robot_exp_group': 'social_robot_experience'
}
demographics = list(demographic_map.keys())

# Create a copy of df with readable demographic column names
df_demo = df.rename(columns=demographic_map)

# For categorical demographics: show counts and proportions
demographic_profiles = []
for col in demographic_map.values():
    if pd.api.types.is_numeric_dtype(df_demo[col]) and df_demo[col].nunique() > 10:
        # Treat as continuous (e.g., age)
        stats = df_demo.groupby('cluster')[col].agg(['mean', 'std', 'min', 'max', 'count'])
        stats.columns = [f"{col}_{stat}" for stat in stats.columns]
        demographic_profiles.append(stats)
    else:
        # Treat as categorical
        counts = df_demo.groupby('cluster')[col].value_counts(dropna=False).unstack(fill_value=0)
        props = counts.div(counts.sum(axis=1), axis=0)
        counts.columns = [f"{col}_count_{c}" for c in counts.columns]
        props.columns = [f"{col}_prop_{c}" for c in props.columns]
        demographic_profiles.append(counts)
        # demographic_profiles.append(props)

# Concatenate all demographic summaries horizontally
demographic_summary = pd.concat(demographic_profiles, axis=1)
demographic_summary.to_csv('results/cluster_profiles_demographics.csv')

print("\nCluster demographic summary (see results/cluster_profiles_demographics.csv):")
print(demographic_summary.head())


### Chi squared test
# Crosstab: early_training (Yes/No) by cluster
crosstab = pd.crosstab(df['early_training'], df['cluster'])
print("\nCrosstab (early_training vs. cluster):")
print(crosstab)

# Chi-square test
chi2, p, dof, expected = chi2_contingency(crosstab)
print(f"\nChi-square: {chi2:.2f}, p-value: {p:.4f}, dof: {dof}")

# Visualize the distribution (proportions)
crosstab_prop = crosstab.div(crosstab.sum(axis=1), axis=0)
crosstab_prop.T.plot(kind='bar', figsize=(8,5))
plt.title('Cluster Distribution by Early Musical Training')
plt.xlabel('Cluster')
plt.ylabel('Proportion within Group')
plt.legend(title='Early Musical Training')
plt.tight_layout()
plt.savefig('plots/cluster_distribution_by_early_training.png')



### Power and Effect Size
def cramers_v(confusion_matrix):
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    return np.sqrt(phi2 / min(k-1, r-1))

cramers_v_value = cramers_v(crosstab)
print(f"Cramér's V: {cramers_v_value:.3f}")

effect_size = cramers_v_value
nobs = crosstab.sum().sum()
alpha = 0.05
power = GofChisquarePower().solve_power(effect_size=effect_size, nobs=nobs, alpha=alpha, n_bins=crosstab.shape[1])
print(f"Power for chi-square test: {power:.3f}")