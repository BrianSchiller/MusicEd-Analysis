# Social Robots in Music Education: Analysis Pipeline

This repository contains all code and resources for the analysis of our survey on attitudes towards social robots in music education.

## Folder Structure

- `data/`  
  Raw and processed data files.  
  - Place the original survey export as `data/data.csv` (tab-separated, UTF-16).
  - Processed data will be saved as `data/processed_data.csv`.

- `results/`  
  All statistical results, tables, and exported CSVs.

- `plots/`  
  All generated figures and visualizations.

- `const.py`  
  Contains scale definitions for subscales.

- `preprocess.py`  
  Cleans the raw data, applies exclusion criteria, and creates derived variables (e.g., early musical training, robot experience group).  
  **Run this first** to generate `processed_data.csv`.

- `rq1.py`, `rq2.py`, `rq3.py`, `rq4.py`  
  Scripts for each research question:
  - **RQ1:** Early musical training and attitudes
  - **RQ2:** Robot experience and attitudes
  - **RQ3:** Interaction of early training and robot experience
  - **RQ4:** Cluster analysis and demographic profiling

## How to Run

1. **Install dependencies**  
   Make sure you have Python 3.8+ and the following packages:
   ```
   pip install pandas numpy matplotlib seaborn scikit-learn statsmodels scipy
   ```

2. **Prepare data**  
   - Place your raw survey export as `data/data.csv` (tab-separated, UTF-16 encoding).

3. **Preprocess**  
   ```
   python preprocess.py
   ```
   This will create `data/processed_data.csv` and print summary statistics.

4. **Run analyses**  
   For each research question, run the corresponding script, e.g.:
   ```
   python rq1.py
   python rq2.py
   python rq3.py
   python rq4.py
   ```
   Results and plots will be saved in the `results/` and `plots/` folders.

---
**Happy analyzing!**
