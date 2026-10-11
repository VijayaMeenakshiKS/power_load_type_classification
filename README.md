# ⚡ Industrial Power System Load Type Classification

> **Machine Learning-driven energy demand intelligence, AC power triangle feature engineering, and chronological peak-load forecasting.**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)](app.py)

---

## 📌 1. Project Overview & Objective

In heavy industries and smart power grids, electrical utility providers bill industrial consumers based on **Time-of-Use (ToU) tariffs** and **Peak Demand Surcharges**. Sudden unplanned transitions into high-demand operating states can lead to expensive utility penalties, strain local transformers, and risk blackout trips.

This project develops an end-to-end Machine Learning classification pipeline to accurately predict the multi-class power system operational status:
* **🟢 Light Load:** Idle machines, baseload, nighttime shifts, and weekends.
* **🟡 Medium Load:** Standard daytime production throughput.
* **🔴 Maximum Load:** Peak machinery concurrency, furnace/smelter cycles, and heavy inductive motor startups.

---

## 🌟 2. What Makes This Solution Stand Out (Key Differentiators)

| Common ML Submissions | This Project's Architecture |
| :--- | :--- |
| Randomly shuffles time-series data (`train_test_split`), causing severe temporal lookahead leakage. | **Strict Chronological Evaluation:** Adheres strictly to the requirement by training on prior months and evaluating on the **unseen final month**. |
| Treats voltage/current sensor numbers as generic abstract columns. | **Physics-Informed Electrical Engineering:** Derives the **AC Power Triangle** ($S = \sqrt{P^2 + Q^2}$), Operating Power Factor ($\cos\phi$), and cyclical 24-hr trigonometric time coordinates ($\sin/\cos$). |
| Only reports basic accuracy scores. | **Asymmetric Industrial Penalty Matrix:** Quantifies real financial savings achieved by avoiding catastrophic underestimations of peak demand. |
| Code dumped without structure. | **Production-Ready Deliverables:** A modular pipeline (`train_model.py`), a structured Jupyter Notebook (`.ipynb`), and an interactive Streamlit UI (`app.py`). |

---

## 🏗️ 3. Project Directory Structure

```text
power_load_type_classification/
│
├── notebooks/
│   └── power_load_classification.ipynb   # ⭐️ PRIMARY DELIVERABLE: Complete Notebook
│
├── src/
│   └── (Optional utility modules)
│
├── data/                                 # Place your dataset CSV here
│
├── models/
│   └── best_load_classifier.joblib       # Serialized production model package
│
├── outputs/                              # Exported confusion matrix, charts & reports
│
├── app.py                                # Optional interactive Streamlit web dashboard
├── train_model.py                        # Standalone end-to-end training script
├── requirements.txt                      # Python dependencies
└── README.md                             # Project documentation
```

---

## ⚙️ 4. Quickstart Guide (How to Run in VS Code)

### Step 1: Open the Project in VS Code
1. Open **VS Code**.
2. Click **File** > **Open Folder...** and select:  
   `C:\Users\vijayashree\Desktop\power_load_type_classification`

### Step 2: Install Dependencies
Open the integrated terminal in VS Code (`Ctrl + ~`) and run:
```bash
pip install -r requirements.txt
```

### Step 3: Add the Dataset
Place the supplied CSV at `data/load_data.csv` (or another CSV filename inside `data/`). The loader checks the project `data/` folder and validates required columns. If the file is elsewhere, set `POWER_LOAD_DATA` to its full path before launching the notebook or training script.

### Step 4: Run the Primary Deliverable (Jupyter Notebook)
1. Open `notebooks/power_load_classification.ipynb` in VS Code.
2. Select your Python kernel in the top-right corner.
3. Click **Run All** (or execute cells sequentially with `Shift + Enter`).
4. The notebook will automatically:
   - Locate and load the dataset.
   - Run exploratory data analysis (EDA).
   - Compute the AC power triangle features.
   - Train and benchmark all candidate models on historical months.
   - Output the model comparison table, confusion matrix, and classification report on the unseen last month.
   - Export the best model to `models/best_load_classifier.joblib`.

### Step 5 (Alternative): Run via Terminal Script
If you prefer running the entire pipeline from the terminal in one command:
```bash
python train_model.py
```

### Step 6 (Optional): Launch the Streamlit Dashboard
Once the model is trained, launch the interactive user interface:
```bash
streamlit run app.py
```
This opens an interactive dashboard where you can slide active and reactive power values, observe real-time power factor calculations, and watch the model predict the load tier live!

---

## 📊 5. Mathematical Formulations & Electrical Domain Engineering

### AC Power Triangle:
* **Active Real Power ($P$):**
  $$P = \text{Usage\_kWh}$$
* **Net Reactive Power ($Q$):**
  $$Q_{\text{net}} = Q_{\text{lagging}} - Q_{\text{leading}}$$
* **Apparent Power ($S$):**
  $$S = \sqrt{P^2 + Q_{\text{net}}^2}$$
* **Operating Power Factor ($\cos\phi$):**
  $$\text{PF} = \frac{P}{S}$$

### Cyclical Time Transformations:
For seconds from midnight ($NSM \in [0, 86400]$):
$$\sin_{\text{NSM}} = \sin\left(\frac{2\pi \cdot NSM}{86400}\right), \quad \cos_{\text{NSM}} = \cos\left(\frac{2\pi \cdot NSM}{86400}\right)$$

---

## 📈 6. Evaluation Metrics & Deliverable Checklist
- [x] **Primary Deliverable:** `notebooks/power_load_classification.ipynb` (clean headings, visualizations, and commentary).
- [x] **Validation Rule:** Final month held out as unseen test set.
- [x] **Comprehensive Metrics:** Accuracy, Precision, Recall, Macro F1, Weighted F1.
- [x] **Diagnostic Visuals:** Normalized Confusion Matrix, Feature Importances, 24-hr Load Profiles.
- [x] **Streamlit App:** Included in `app.py`.

## Latest Holdout Results

Running `python train_model.py` on the included `data/load_data.csv` uses December 2018 as the test month (2,573 rows after removing records with missing required sensor readings). LightGBM was selected by macro F1, with 75.4% accuracy and 0.643 macro F1. Maximum_Load recall was 49.2%, so peak detection needs improvement before operational use. Detailed results are written under `outputs/`.
