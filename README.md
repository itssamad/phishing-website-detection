# Phishing Website Detection — Machine Learning Pipeline

A machine learning pipeline that classifies websites as phishing or legitimate based on structural and behavioural URL features, and separately mines the dataset to find which features most strongly indicate phishing.

## What it does

1. **Classification** — Loads the dataset in PySpark, standardises the features, and trains a Linear SVM to classify each site as phishing or legitimate. Evaluates the model with accuracy, precision, recall, F1-score, a full classification report, and a confusion matrix (saved as `svm_confusion_matrix.png`).
2. **Association rule mining** — Converts key features (IP address usage, SSL state, URL anchors, domain age, DNS record presence, etc.) into transactions and mines association rules that co-occur with a phishing result, comparing two different algorithms:
   - **Apriori** (via `mlxtend`)
   - **FP-Growth** (via PySpark MLlib)
3. **Comparison** — Times and compares both algorithms on itemsets found, rules generated, and runtime, exporting the results to `association_comparison.csv` along with runtime/rule-count comparison charts.

## Dataset

`dataset.csv` — the UCI Phishing Websites dataset, 11,000+ rows, each a website described by 30 binary/categorical features (e.g. use of an IP address instead of a domain, SSL certificate state, URL length, presence of a prefix/suffix, domain age) with a ground-truth label of phishing or legitimate.

## Tech stack

Python · PySpark (MLlib) · scikit-learn · mlxtend · pandas · matplotlib

## Getting started

```bash
pip install -r requirements.txt
```

You'll also need a JDK 17 installed locally and `JAVA_HOME` set, since PySpark runs on the JVM.

```bash
python phishing_data_mining.py
```

## Results

The SVM classifier and both association-rule-mining algorithms run end-to-end on the full dataset, printing dataset diagnostics, model performance metrics, and the strongest phishing-associated rules directly to the console, with plots saved alongside the script.
