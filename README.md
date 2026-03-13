# Give Me Some Credit: Predictive Credit Risk Modeling for Financial Institutions

**By: Brya Patterson, Jessica Grubb, Tim Jooo** **Spring 2025, University of Washington, MS Information Management (MSIM)**

---

## Project Overview

This project addresses the "Credit Scoring" problem for financial institutions by predicting the probability that a borrower will experience serious delinquency (90+ days past due) within a two-year window. Using a dataset of 120,000+ borrowers, our team developed a machine learning pipeline that prioritizes high-risk identification over simple binary accuracy to better mitigate lender loss.

### Guiding Research Questions

* What is the probability of a given individual defaulting on their loan within the next two years?
* Which financial indicators are the strongest predictors of loan default?
* Can we identify specific customer segments with a high likelihood of default?
* How do changes in a borrower's financial situation impact their predicted risk?
* What are the key differences in behavior between borrowers who default and those who do not?

## The Data

The analysis utilizes historical data from the Kaggle "Give Me Some Credit" competition, featuring 11 key variables.

* **Pre-processing:** Handled significant missing values in `MonthlyIncome` (29,731) and `NumberOfDependents` (3,924) through listwise deletion to ensure data integrity, resulting in a final dataset of **120,269** records.
* **Data Split:** Utilized an 80/20 train-test split to validate model performance on unseen data.
* **Class Imbalance:** Addressed a severe **13:1 class imbalance** (6.7% default rate) using cost-sensitive learning techniques.

## Technical Implementation

### 1. Predictive Modeling

We utilized a **Balanced Random Forest Classifier** to handle the minority class effectively.

* **Recall for Defaulters:** 61% (successfully flagging high-risk individuals).
* **Precision for Non-Defaulters:** 96% (high confidence in automated approvals).
* **Key Technique:** Implemented `class_weight='balanced'` to prioritize catching defaults over simple accuracy.

### 2. Customer Segmentation

Using **K-Means Clustering** and the Elbow Method, we identified four distinct risk personas:

* **Stable Seniors:** Characterized by higher age and clean history; lowest default rate (~4.3%).
* **High-Risk Delinquents:** Characterized by frequent late payments and high utilization (~55.2% default rate).

## Key Insights

* **Behavior > Debt:** Past payment history (specifically 90+ days late) is a **1,375% stronger predictor** of default than current debt levels.
* **The Age Protective Factor:** Risk steadily decreases with age, with a significant "stability drop-off" occurring after age 55.
* **Income Sensitivity:** A $2,000 increase in monthly income was found to reduce default probability by up to **97%** in high-risk simulations.
* **The Utilization Myth:** Credit utilization and Debt Ratio had surprisingly low predictive power compared to behavioral history.

## Repository Structure

* `IMT574_FINAL_Project.ipynb`: Full exploratory data analysis (EDA), visualizations, and model training.
* `GiveMeSomeCredit folder`: Access all necessary files for analysis.
* `README.md`: Project summary and key findings.

## How to Use

1. **Clone the repo:** 
```
bash 
git clone https://github.com/bryap/give-me-some-credit
```


2. **Install dependencies:** 
```
bash
pip install pandas scikit-learn matplotlib seaborn
```


3. **Run the analysis:** Open the Jupyter Notebook to view the full pipeline.

---

*For a full technical report and executive summary of this project, please visit https://bryap.github.io/datasci_credit.html*