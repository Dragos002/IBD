import json

# Define the structure of the new Multi-CSV Notebook
notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Mobile Malware Classification - Multi-View Data Fusion\n",
    "\n",
    "**Project:** Malware Detection using Hybrid Features (Static + Dynamic)\n",
    "**Platform:** Apache Spark (PySpark)\n",
    "\n",
    "This notebook implements a complex pipeline that ingests multiple CSV sources, fuses them into a single feature set, and trains ML models.\n",
    "\n",
    "### Data Sources:\n",
    "1. **Static Analysis:** Raw attributes (Intents, Activities) -> Processed via TF-IDF.\n",
    "2. **Dynamic Syscalls:** System call frequencies -> Processed via VectorAssembler.\n",
    "3. **Permissions/Binders:** High-level actions (ACCESS_PERSONAL_INFO) -> Processed via VectorAssembler.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 1. Setup & Imports\n",
    "!pip install pyspark pandas seaborn matplotlib numpy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pyspark.sql.functions as F\n",
    "from pyspark.sql import SparkSession\n",
    "from pyspark.sql.types import StringType, IntegerType\n",
    "from pyspark.ml import Pipeline\n",
    "from pyspark.ml.feature import VectorAssembler, StringIndexer, StandardScaler, Tokenizer, HashingTF, IDF, SQLTransformer\n",
    "from pyspark.ml.classification import RandomForestClassifier, LogisticRegression\n",
    "from pyspark.ml.evaluation import MulticlassClassificationEvaluator\n",
    "\n",
    "# Initialize Spark with more memory for the join operations\n",
    "spark = SparkSession.builder \\\n",
    "    .appName(\"Malware_MultiSource_Fusion\") \\\n",
    "    .config(\"spark.executor.memory\", \"8g\") \\\n",
    "    .config(\"spark.driver.memory\", \"8g\") \\\n",
    "    .config(\"spark.sql.autoBroadcastJoinThreshold\", -1) \\ # Disable broadcast to force sort-merge join if needed\n",
    "    .getOrCreate()\n",
    "\n",
    "print(f\"Spark Active. Version: {spark.version}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 2. Multi-Source Ingestion\n",
    "\n",
    "# Define file paths (Make sure these match your renamed files)\n",
    "PATH_STATIC = \"static.csv\"         # The big CSV with 'a:targetActivity...'\n",
    "PATH_SYSCALLS = \"syscalls.csv\"     # The numeric syscall frequencies\n",
    "PATH_PERMS = \"permissions.csv\"     # The ACCESS_PERSONAL_INFO file\n",
    "\n",
    "print(\"Reading CSVs...\")\n",
    "# We force reading raw unformatted text for static to handle the weird encoding\n",
    "df_static = spark.read.text(PATH_STATIC).withColumnRenamed(\"value\", \"raw_static_text\")\n",
    "\n",
    "# Read numeric/structured files normally\n",
    "df_syscalls = spark.read.csv(PATH_SYSCALLS, header=True, inferSchema=True)\n",
    "df_perms = spark.read.csv(PATH_PERMS, header=True, inferSchema=True)\n",
    "\n",
    "print(f\"Static Rows: {df_static.count()}\")\n",
    "print(f\"Syscalls Rows: {df_syscalls.count()}\")\n",
    "print(f\"Permissions Rows: {df_perms.count()}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 3. Data Alignment & Fusion (Merging the CSVs)\n",
    "\n",
    "# Since we don't have a common ID column, we assume row alignment.\n",
    "# We add a sequential ID to each dataframe to join them.\n",
    "\n",
    "def add_index(df):\n",
    "    # Adds a monotonically increasing id to allow joining\n",
    "    return df.withColumn(\"row_id\", F.monotonically_increasing_id())\n",
    "\n",
    "df_static = add_index(df_static)\n",
    "df_syscalls = add_index(df_syscalls)\n",
    "df_perms = add_index(df_perms)\n",
    "\n",
    "# Join DataFrames\n",
    "print(\"Joining DataFrames...\")\n",
    "df_merged = df_syscalls.join(df_perms, \"row_id\").join(df_static, \"row_id\")\n",
    "\n",
    "# Drop the temp row_id\n",
    "df_merged = df_merged.drop(\"row_id\")\n",
    "\n",
    "# FIND THE LABEL/CLASS COLUMN\n",
    "# Usually, the class is in one of the files. Let's try to auto-detect it.\n",
    "columns = df_merged.columns\n",
    "possible_labels = ['class', 'Class', 'label', 'Label', 'family', 'type']\n",
    "target_col = None\n",
    "\n",
    "for c in columns:\n",
    "    if c in possible_labels:\n",
    "        target_col = c\n",
    "        break\n",
    "\n",
    "# Fallback: If no label found, user needs to specify. For now, let's assume 'Class' exists.\n",
    "if target_col is None:\n",
    "    print(\"WARNING: Could not find a column named 'Class'. Using the last column as target.\")\n",
    "    target_col = columns[-2] # -1 might be raw_static_text, so taking -2 or checking\n",
    "else:\n",
    "    print(f\"Target Column identified: {target_col}\")\n",
    "\n",
    "# Clean up Target: If it's string, we need to index it later. If numeric, keep it.\n",
    "df_merged.select(target_col).show(5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 4. Feature Engineering Pipeline\n",
    "\n",
    "# --- A. Static Features (Text Handling) ---\n",
    "# The static file has content like: a:targetActivity+AD0-'com.mplus...'\n",
    "# We treat this as a 'document' and calculate TF-IDF.\n",
    "\n",
    "tokenizer = Tokenizer(inputCol=\"raw_static_text\", outputCol=\"static_words\")\n",
    "hashingTF = HashingTF(inputCol=\"static_words\", outputCol=\"static_features_raw\", numFeatures=1000)\n",
    "idf = IDF(inputCol=\"static_features_raw\", outputCol=\"static_features_vec\")\n",
    "\n",
    "# --- B. Dynamic & Permission Features (Numeric) ---\n",
    "# Collect all numeric columns from syscalls and perms\n",
    "ignore_cols = [\"raw_static_text\", \"static_words\", \"static_features_raw\", \"static_features_vec\", target_col, \"row_id\"]\n",
    "numeric_cols = [c for c in df_merged.columns if c not in ignore_cols]\n",
    "\n",
    "# Assemble numeric features\n",
    "assembler_numeric = VectorAssembler(inputCols=numeric_cols, outputCol=\"numeric_features_vec\", handleInvalid=\"skip\")\n",
    "\n",
    "# --- C. Combine Everything ---\n",
    "assembler_final = VectorAssembler(inputCols=[\"static_features_vec\", \"numeric_features_vec\"], outputCol=\"features\")\n",
    "\n",
    "# --- D. Label Indexing ---\n",
    "label_indexer = StringIndexer(inputCol=target_col, outputCol=\"label\").setHandleInvalid(\"skip\")\n",
    "\n",
    "# Create the Pipeline\n",
    "pipeline = Pipeline(stages=[tokenizer, hashingTF, idf, assembler_numeric, assembler_final, label_indexer])\n",
    "\n",
    "print(\"Running Feature Pipeline (This may take a minute due to Text Processing)...\")\n",
    "model_prep = pipeline.fit(df_merged)\n",
    "final_data = model_prep.transform(df_merged).select(\"features\", \"label\")\n",
    "\n",
    "final_data.show(5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 5. Train & Evaluate Models\n",
    "\n",
    "# Split Data\n",
    "train, test = final_data.randomSplit([0.7, 0.3], seed=42)\n",
    "\n",
    "# Models to test\n",
    "rf = RandomForestClassifier(labelCol=\"label\", featuresCol=\"features\", numTrees=50)\n",
    "lr = LogisticRegression(labelCol=\"label\", featuresCol=\"features\", maxIter=10)\n",
    "\n",
    "print(\"Training Random Forest...\")\n",
    "rf_model = rf.fit(train)\n",
    "rf_preds = rf_model.transform(test)\n",
    "\n",
    "print(\"Training Logistic Regression...\")\n",
    "lr_model = lr.fit(train)\n",
    "lr_preds = lr_model.transform(test)\n",
    "\n",
    "# Evaluation Function\n",
    "def evaluate_model(predictions, model_name):\n",
    "    acc_eval = MulticlassClassificationEvaluator(metricName=\"accuracy\")\n",
    "    f1_eval = MulticlassClassificationEvaluator(metricName=\"f1Weighted\")\n",
    "    \n",
    "    acc = acc_eval.evaluate(predictions)\n",
    "    f1 = f1_eval.evaluate(predictions)\n",
    "    \n",
    "    print(f\"Results for {model_name}:\")\n",
    "    print(f\"  Accuracy: {acc:.4f}\")\n",
    "    print(f\"  F1-Score: {f1:.4f}\")\n",
    "    return acc, f1\n",
    "\n",
    "acc_rf, f1_rf = evaluate_model(rf_preds, \"Random Forest\")\n",
    "acc_lr, f1_lr = evaluate_model(lr_preds, \"Logistic Regression\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 6. Visual Comparison\n",
    "import pandas as pd\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "data = {\n",
    "    'Model': ['Random Forest', 'Random Forest', 'Logistic Reg', 'Logistic Reg'],\n",
    "    'Metric': ['Accuracy', 'F1-Score', 'Accuracy', 'F1-Score'],\n",
    "    'Score': [acc_rf, f1_rf, acc_lr, f1_lr]\n",
    "}\n",
    "\n",
    "df_viz = pd.DataFrame(data)\n",
    "\n",
    "plt.figure(figsize=(8,5))\n",
    "sns.barplot(x='Model', y='Score', hue='Metric', data=df_viz)\n",
    "plt.title(\"Model Comparison: Hybrid Features\")\n",
    "plt.ylim(0, 1.1)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Stop Spark\n",
    "spark.stop()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

# Save to file
filename = "Malware_Hybrid_Fusion.ipynb"
with open(filename, 'w') as f:
    json.dump(notebook_content, f)

print(f"✅ Notebook generated: {filename}")