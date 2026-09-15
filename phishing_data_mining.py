import time
import pandas as pd
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, concat_ws, array, lit
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LinearSVC
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.fpm import FPGrowth
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Run this script from the repo root so dataset.csv resolves correctly,
# and make sure JAVA_HOME points at a JDK 17 install (required by PySpark).

spark = SparkSession.builder.appName("Phishing Detection").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

df = spark.read.csv("dataset.csv", header=True, inferSchema=True)

print("DATASET OVERVIEW")
print("Rows:", df.count())
print("Columns:", len(df.columns))

print("\nMISSING VALUES")
missing = [(c, df.filter(col(c).isNull()).count()) for c in df.columns]
print(pd.DataFrame(missing, columns=["Column", "Missing Values"]))

print("\nCLASS DISTRIBUTION")
df.groupBy("Result").count().show()

df = df.withColumn("label", when(col("Result") == -1, 1).otherwise(0))
df.groupBy("label").count().show()

remove_cols = ["index", "Result", "label"]
feature_cols = [c for c in df.columns if c not in remove_cols]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features")
df_vector = assembler.transform(df)

scaler = StandardScaler(
    inputCol="raw_features",
    outputCol="features",
    withMean=True,
    withStd=True
)

df_scaled = scaler.fit(df_vector).transform(df_vector)

train_data, test_data = df_scaled.randomSplit([0.8, 0.2], seed=42)

print("Training rows:", train_data.count())
print("Testing rows: ", test_data.count())

svm = LinearSVC(
    featuresCol="features",
    labelCol="label",
    maxIter=50,
    regParam=0.1
)

svm_model = svm.fit(train_data)
predictions = svm_model.transform(test_data)

def evaluate_model(predictions):
    results = {}

    for metric in ["accuracy", "weightedPrecision", "weightedRecall", "f1"]:
        evaluator = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName=metric
        )
        results[metric] = evaluator.evaluate(predictions)

    return results

metrics = evaluate_model(predictions)

print("\nSVM RESULTS")
print(f"Accuracy:  {metrics['accuracy']:.4f}")
print(f"Precision: {metrics['weightedPrecision']:.4f}")
print(f"Recall:    {metrics['weightedRecall']:.4f}")
print(f"F1-score:  {metrics['f1']:.4f}")

pred_pd = predictions.select("label", "prediction").toPandas()
cm = confusion_matrix(pred_pd["label"], pred_pd["prediction"])

print("\nCONFUSION MATRIX")
print(cm)

print("\nCLASSIFICATION REPORT")
print(
    classification_report(
        pred_pd["label"],
        pred_pd["prediction"],
        target_names=["Legitimate", "Phishing"]
    )
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Legitimate", "Phishing"]
)

disp.plot()
plt.title("SVM Confusion Matrix")
plt.savefig("svm_confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()

assoc_features = [
    "having_IPhaving_IP_Address",
    "SSLfinal_State",
    "URL_of_Anchor",
    "Request_URL",
    "age_of_domain",
    "DNSRecord",
    "Prefix_Suffix",
    "HTTPS_token"
]

df_assoc = df
item_cols = []

for c in assoc_features:
    item_col = c + "_item"
    df_assoc = df_assoc.withColumn(
        item_col,
        concat_ws("=", lit(c), col(c).cast("string"))
    )
    item_cols.append(item_col)

df_assoc = df_assoc.withColumn(
    "Result_item",
    when(col("Result") == -1, lit("Result=Phishing")).otherwise(lit("Result=Legitimate"))
)

item_cols.append("Result_item")

df_items = df_assoc.withColumn(
    "items",
    array(*[col(c) for c in item_cols])
)

transactions = df_items.select("items").toPandas()["items"].tolist()

min_support = 0.25
min_confidence = 0.80

te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
transaction_df = pd.DataFrame(te_array, columns=te.columns_)

start_time = time.time()

freq_itemsets = apriori(
    transaction_df,
    min_support=min_support,
    use_colnames=True
)

ap_rules = association_rules(
    freq_itemsets,
    metric="confidence",
    min_threshold=min_confidence
)

apriori_time = time.time() - start_time

ap_rules = ap_rules.sort_values("confidence", ascending=False)

ap_phishing = ap_rules[
    ap_rules["consequents"].astype(str).str.contains("Result=Phishing")
]

print("\nAPRIORI RESULTS")
print("Frequent itemsets:", len(freq_itemsets))
print("Association rules:", len(ap_rules))
print("Phishing rules:   ", len(ap_phishing))
print("Runtime (s):      ", round(apriori_time, 4))

print("\nAPRIORI PHISHING RULES")
print(
    ap_phishing[
        ["antecedents", "consequents", "support", "confidence", "lift"]
    ].to_string()
)

start_time = time.time()

fp_growth = FPGrowth(
    itemsCol="items",
    minSupport=min_support,
    minConfidence=min_confidence
)

fp_model = fp_growth.fit(df_items)

fp_time = time.time() - start_time

fp_items_pd = fp_model.freqItemsets.orderBy(col("freq").desc()).toPandas()
fp_rules_pd = fp_model.associationRules.orderBy(col("confidence").desc()).toPandas()

fp_phishing = fp_rules_pd[
    fp_rules_pd["consequent"].astype(str).str.contains("Result=Phishing")
].reset_index(drop=True)

print("\nFP-GROWTH RESULTS")
print("Frequent itemsets:", len(fp_items_pd))
print("Association rules:", len(fp_rules_pd))
print("Phishing rules:   ", len(fp_phishing))
print("Runtime (s):      ", round(fp_time, 4))

print("\nFP-GROWTH PHISHING RULES")
print(
    fp_phishing[
        ["antecedent", "consequent", "support", "confidence", "lift"]
    ].to_string()
)

comparison = pd.DataFrame({
    "Algorithm": ["Apriori (mlxtend)", "FP-Growth (PySpark)"],
    "Min Support": [min_support, min_support],
    "Min Confidence": [min_confidence, min_confidence],
    "Frequent Itemsets": [len(freq_itemsets), len(fp_items_pd)],
    "Rules Generated": [len(ap_rules), len(fp_rules_pd)],
    "Phishing Rules": [len(ap_phishing), len(fp_phishing)],
    "Runtime (s)": [round(apriori_time, 4), round(fp_time, 4)]
})

print("\nCOMPARISON: APRIORI vs FP-GROWTH")
print(comparison.to_string(index=False))

comparison.to_csv("association_comparison.csv", index=False)

plt.figure(figsize=(6, 4))
plt.bar(
    ["Apriori", "FP-Growth"],
    [apriori_time, fp_time]
)
plt.ylabel("Runtime (seconds)")
plt.title("Apriori vs FP-Growth Runtime")
plt.tight_layout()
plt.savefig("runtime_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(6, 4))
plt.bar(
    ["Apriori", "FP-Growth"],
    [len(ap_rules), len(fp_rules_pd)]
)
plt.ylabel("Rules Generated")
plt.title("Apriori vs FP-Growth Rules Generated")
plt.tight_layout()
plt.savefig("rules_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

print("\nAnalysis completed successfully.")

spark.stop()
