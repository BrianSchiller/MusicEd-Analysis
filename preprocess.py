import pandas as pd
import os

# Use the correct file path and delimiter
file_path = 'data/data.csv'  # or 'data.tsv' if you want to rename

# Create folders if they do not exist
for folder in ['plots', 'results', 'data']:
    os.makedirs(folder, exist_ok=True)

# Try reading with utf-16 and tab separator, skipping the second row (labels)
df = pd.read_csv(file_path, sep='\t', encoding='utf-16', skiprows=[1])

# Show first few rows
print("First few rows of the dataset:")
print(df.head(10))
print(df.shape)

# 1. Remove all participants that did not complete the survey
df = df[df['FINISHED'] == 1]

# 2. Remove all participants that did not pass manipulation check
# MC04: TRUE = 1
df = df[df['MC04'] == 1]
print("After removing participants who did not pass manipulation check:")
print(df.shape)

# 3. Remove all participants that did not pass attention checks
# AT02_46: leftmost = 1
df = df[df['AT02_46'] == 1]

# AT02_47: rightmost = 6
df = df[df['AT02_47'] == 6]
print("After removing participants who did not pass attention checks:")
print(df.shape)


# Add early_training column (1 = Yes, 2 = No)
df['early_training'] = df.apply(
    lambda row: 'Yes' if (row['SD05'] == 1 and pd.notna(row['SD06_01']) and row['SD06_01'] <= 11) else 'No',
    axis=1
)

# Add robot_exp_group column (No: 1, Yes: 2/3/4/5/6)
def exp_group(val):
    if val in [1]:
        return 'No'
    elif val in [2, 3, 4, 5, 6]:
        return 'Yes'
    else:
        return pd.NA

df['robot_exp_group'] = df['RS01_01'].apply(exp_group)



# # Did you receive musical education?
# print(df['SD05'].value_counts(dropna=False))


print("What kind of music education did you receive?")
print(df['SD08'].value_counts(dropna=False))

# print("Where did you receive your musical education?")
# print(df['SD09'].value_counts(dropna=False))


# Show all columns that start with 'SD' and aggregate their value counts
sd_columns = [col for col in df.columns if col.startswith('SD')]
# print("\nAggregated value counts for all SD__ columns:")
# for col in sd_columns:
#     print(f"\n{col}:")
#     print(df[col].value_counts(dropna=False))

# If you want to see a quick summary (like describe) for all SD columns:
print("\nSummary statistics for all SD__ columns:")
summary = df[sd_columns].describe(include='all')
print(summary)

# Export the summary table to a CSV file
summary.to_csv('data/SD_columns_summary.csv')

df.to_csv('data/processed_data.csv', index=False)

