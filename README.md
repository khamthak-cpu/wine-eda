# Wine Dataset — Exploratory Data Analysis

An EDA project on the sklearn wine dataset: 178 samples, 13 chemical features, 3 wine classes.

## What I did
- Data quality checks (missing values, duplicates)
- Distribution analysis (histograms, skewness)
- Class comparison (boxplots, groupby)
- Correlation analysis (scatter plots, heatmap)

## Key findings
1. Flavanoids best separate the wine classes (2.98 vs 0.78 mean)
2. total_phenols ↔ flavanoids correlation ≈ 0.9 (redundant features)
3. proline's scale is 100× larger — normalization needed for ML
4. ...

## Tools
Python, pandas, matplotlib, seaborn, scikit-learn

## How to run
pip install pandas matplotlib seaborn scikit-learn scipy
python main.py
