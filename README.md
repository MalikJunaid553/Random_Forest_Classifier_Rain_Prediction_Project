# Weather Forecasting Pipeline: Confronting Class Imbalance in Random Forests

This repository features an end-to-end Machine Learning pipeline that predicts precipitation outcomes using diagnostic meteorological indicators. The project explores **Ensemble Learning** via a **Random Forest Classifier**, exposing the operational pitfalls of the **Accuracy Paradox** when training models on highly skewed dataset balances.

## 📌 Project Overview
The goal of this project is to implement a robust binary classification network capable of forecasting rain events. By analyzing feature behaviors across an imbalanced dataset, the pipeline demonstrates why secondary validation heuristics (Precision, Recall, F1-Score) are mandatory safeguards over baseline accuracy statistics.

### Key Highlights:
* **Ensemble Architecture**: Advanced from standalone decision boundaries to an aggregated bootstrap ensemble using Random Forests.
* **The Accuracy Paradox**: Documented a real-world scenario where a model registers 90% baseline accuracy yet achieves an active F1-score of 0.00 due to severe class skews.
* **Feature Utility Profiles**: Quantified empirical importance weights across multiple shifting weather metrics.
* **Production Serialization Checkpoint**: Validated model preservation loops by dumping and reloading active binary artifacts from storage.

---

## 🔧 Technical Toolkit
* **Language**: Python 3.x
* **Data Manipulation**: `pandas`, `numpy`
* **Machine Learning**: `scikit-learn`
* **Data Visualization**: `matplotlib`
* **Model Serialization**: `pickle`

---

## 📊 Dataset Features & Target Skew
The underlying dataset tracks 2,500 diagnostic weather records consisting of:
* `Temperature`
* `Humidity`
* `Wind_Speed`
* `Cloud_Cover`
* `Pressure`
* `Rain` (Target: String label mapped directly via automated vector matrices to `1` for Rain, `0` for No Rain)

### Class Distribution (Visualized via Pie Chart):
* **No Rain (Class 0)**: 2,186 samples (**87.4%** of dataset)
* **Rain (Class 1)**: 314 samples (**12.6%** of dataset)

---

## 🚀 Pipeline Workflow

### 1. Vectorized Preprocessing & Target Splitting
* Isolated parameters from categorical tracking text labels.
* Transformed text outcomes into mathematical binary arrays via rapid vector maps:
  ```python
  y = np.where(y == "rain", 1, 0)
  ```
* Partitioned features using a standardized **80/20 validation split**.

### 2. Ensemble Training & Validation Metrics
* Configured a restricted `RandomForestClassifier(max_depth=2, random_state=0)` to isolate macro trends.
* **Observed Accuracy**: **90.0%**
* **The Structural Failure**: Because the underlying majority class represents 87.4% of the rows, the shallow forest optimized its global accuracy loss function by categorically classifying all test records as `0`. This resulted in an undefined metric warning with true minority Recall dropping to `0.00`.

### 3. Feature Importance Extraction
* Extracted localized gini splits from the underlying tree ensemble via `clf.feature_importances_`:
  * **Humidity**: **44.6%** (Runaway predictive driver)
  * **Cloud_Cover**: **32.8%**
  * **Temperature**: **19.2%**
  * **Wind_Speed / Pressure**: Combined under **3.5%** structural utility.

### 4. Serialization & Multi-Stage Inference Checks
* Serialized the model parameters to storage using `pickle.dump()`.
* Re-read the file into a deployment-ready mock environment to confirm cross-platform persistence:
  ```python
  with open("random_forest_model.pkl", "rb") as f:
      clf_loaded = pickle.load(f)
  clf_loaded.predict([[24, 90, 3, 70, 985]])
  ```

---

## 📈 Visualizations Included in Code
* **Exploratory Class Breakdown (Pie Chart)**: Highlighting the 87.4% split dynamics.
* **Feature Importance Profiles (Horizontal Bar Chart)**: Ranking dimensional structural value.

---

## 🛠️ How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com[YOUR-USERNAME]/[YOUR-REPO-NAME].git
   cd [YOUR-REPO-NAME]
   ```

2. **Install required dependencies:**
   ```bash
   pip install pandas numpy scikit-learn matplotlib
   ```

3. **Execute the workspace notebook** to generate your `.pkl` model artifacts and verify performance metrics locally.
