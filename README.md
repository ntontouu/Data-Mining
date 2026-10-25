# Decision Trees vs. k-NN for Malicious Website Detection

[![University](https://img.shields.io/badge/University-University%20of%20Ioannina-blue.svg)](https://www.uoi.gr)
[![Department](https://img.shields.io/badge/Department-Informatics%20%26%20Telecommunications-lightgrey.svg)](https://www.cs.uoi.gr/)
[![Course](https://img.shields.io/badge/Course-Data%20Mining-orange.svg)]()
[![Libraries](https://img.shields.io/badge/Libraries-Scikit--Learn%20%7C%20Pandas%20%7C%20Seaborn-blueviolet.svg)]()

A robust Python implementation comparing the performance of **Decision Trees (DT)** and **k-Nearest Neighbors (k-NN)** machine learning algorithms. The project evaluates their efficiency in identifying and classifying malicious (phishing) websites based on a given dataset.

---

## 📚 Table of Contents
1. [Overview](#-overview)
2. [Theoretical Background](#-theoretical-background)
3. [Project Structure & Code Architecture](#-project-structure--code-architecture)
4. [Experimental Results & Analysis](#-experimental-results--analysis)
5. [Getting Started & Installation](#-getting-started--installation)
6. [Author & Credits](#-author--credits)

---

## 🔍 Overview
Identifying phishing websites is a critical cybersecurity challenge. This project leverages supervised machine learning to classify URLs as either legitimate (1) or phishing (-1). The dataset is slightly unbalanced (56.8% legitimate, 43.2% phishing). The implementation utilizes a 10-fold cross-validation method to ensure generalization and avoid overfitting.

---

## 📐 Theoretical Background

### Preprocessing & Normalization
- **Decision Trees:** Do not require feature scaling, as they rely on node-splitting rules rather than spatial distance.
- **k-NN:** Highly sensitive to distance metrics. MinMax scaling is applied to map features to the $[0, 1]$ range, preventing features with large magnitudes from dominating the Euclidean distance calculations.

### Hyperparameter Tuning
- **Max Leaf Nodes (DT):** Values tested range from 5 to 50 to balance underfitting and overfitting.
- **k-Value (k-NN):** Tested multiple values (1, 3, 5, 7, 9, 11). Lower values introduce noise sensitivity, while higher values offer stability. The optimal balance was observed at $k=5$.

---

## 🛠️ Project Structure & Code Architecture

| File | Description |
| :--- | :--- |
| **`dt_alg.py`** | Standalone implementation of the Decision Tree Classifier. |
| **`knn_alg.py`** | Standalone implementation of the k-NN Classifier. |
| **`combo_dt_knn_algs.py`** | Combined script featuring data preprocessing, model training for both algorithms, cross-validation, and confusion matrix visualization. |
| **`csv_result-Training Dataset.csv`** | The underlying dataset containing the features and target labels (Result). |

---

## 📊 Experimental Results & Observations

Based on 10-fold cross-validation and confusion matrix analysis:

- **Decision Tree Performance:**
  - **Mean Accuracy:** ~0.961 (96.1%)
  - **F1 Score:** ~0.963
  - **Observation:** Exhibited superior performance, with very low False Positives (48) and False Negatives (44).

- **k-NN Performance:**
  - **Mean Accuracy:** ~0.932 (93.2%)
  - **F1 Score:** ~0.946
  - **Observation:** Solid performance but slightly lower than DT. Misclassifications were slightly higher (67 False Positives, 69 False Negatives).

**Conclusion:** The Decision Tree algorithm proved to be more capable and precise in categorizing the data for this specific problem domain.

---

## 🚀 Getting Started & Execution
1. Install requirements: `pip install pandas scikit-learn matplotlib seaborn`
2. Run the combined script: `python combo_dt_knn_algs.py`
3. Observe the terminal output for classification reports and the GUI windows for Confusion Matrices.

---

## 📝 Author & Academic Use
- **Student:** Maria Alexandra Ndontou (ΜΑΡΙΑ ΑΛΕΞΑΝΔΡΑ ΝΤΟΝΤΟΥ)
- **Student ID:** 2799
- **Institution:** University of Ioannina — Department of Informatics and Telecommunications
- **Course:** Data Mining (Εξόρυξη Δεδομένων) — 1st Semester Project
