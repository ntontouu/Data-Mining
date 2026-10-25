# Custom k-Means Clustering Implementation

[![University](https://img.shields.io/badge/University-University%20of%20Ioannina-blue.svg)](https://www.uoi.gr)
[![Department](https://img.shields.io/badge/Department-Informatics%20%26%20Telecommunications-lightgrey.svg)](https://www.cs.uoi.gr/)
[![Course](https://img.shields.io/badge/Course-Data%20Mining-orange.svg)]()
[![Libraries](https://img.shields.io/badge/Libraries-Numpy%20%7C%20Matplotlib-blueviolet.svg)]()

A from-scratch Python implementation of the **k-Means clustering** algorithm. This unsupervised machine learning project focuses on mathematically categorizing unlabelled data into predefined groups based on spatial feature similarity and minimizing the Sum of Squared Errors (SSE).

---

## 📚 Table of Contents
1. [Overview](#-overview)
2. [Theoretical Background](#-theoretical-background)
3. [Project Structure & Code Architecture](#-project-structure--code-architecture)
4. [Experimental Results & Analysis](#-experimental-results--analysis)
5. [Getting Started & Execution](#-getting-started--execution)
6. [Author & Credits](#-author--credits)

---

## 🔍 Overview
Unsupervised learning operates without pre-labelled training data. The primary objective here is to organize raw data points according to similarities, patterns, and variances. The k-Means algorithm achieves this by iteratively shifting cluster centroids to minimize the variance within each cluster.

---

## 📐 Theoretical Background

### Distance Metric
The similarity between data points is calculated using the **Euclidean Distance** (implemented via NumPy's L2 norm: `np.linalg.norm`):
$$d(x, y) = \sqrt{\sum (x_i - y_i)^2}$$

### The k-Means Lifecycle
1. **Initialization:** Randomly assign $K$ centroids within the data space.
2. **Assignment:** Calculate the distance from each point to all centroids; assign the point to the closest one.
3. **Update:** Recalculate the position of each centroid by taking the mean of all points assigned to that cluster.
4. **Convergence:** Repeat until centroids stabilize (difference between iterations is $< threshold$) or `max_iters` is reached.

### Synthetic Data Generation
To test the algorithm, 3 distinct Gaussian distributions are generated using `np.random.multivariate_normal` (50 points each, 150 total), defining unique spatial clusters controlled by mean ($\mu$) and covariance ($\Sigma$) matrices.

---

## 🛠️ Project Structure & Code Architecture

| File | Description |
| :--- | :--- |
| **`KMEANS_ALG.py`** | The core script containing the `euclidean_distance` function, the `mykmeans` algorithm, data generation, and plotting logic. |
| **`.vscode/settings.json`** | Editor-specific settings for the workspace. |

---

## 📊 Experimental Results & Observations

- **Clustering Performance:** The custom algorithm successfully identified the 3 synthetic clusters ($K=3$) mapped in a 2D plane.
- **Convergence Rate:** The algorithm required only **5 iterations** to reach an optimal state.
- **SSE Tracking:** A plot of the Sum of Squared Errors (SSE) across iterations shows a steep drop during the first 3 iterations, completely stabilizing by the 4th iteration. This indicates an efficient convergence towards the local/global minimum variance without unnecessary computational overhead.

---

## 🚀 Getting Started & Execution
1. Install requirements: `pip install numpy matplotlib`
2. Run the script: `python KMEANS_ALG.py`
3. The algorithm will display an animated plot updating the cluster centroids in real-time. Close the first window to view the final SSE convergence graph.

---

## 📝 Author & Academic Use
- **Student:** Maria Alexandra Ndontou (ΜΑΡΙΑ ΑΛΕΞΑΝΔΡΑ ΝΤΟΝΤΟΥ)
- **Student ID:** 2799
- **Institution:** University of Ioannina — Department of Informatics and Telecommunications
- **Course:** Data Mining (Εξόρυξη Δεδομένων) — 2nd Semester Project
