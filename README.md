# Uganda Integrated Care Intelligence System (UGICIS)

## A Missingness-Aware Multi-Task Clinical Intelligence System for HIV-Hypertension Risk Prediction and Clinical Pathway Support in Ugandan Healthcare Environments

---

# Author Information

**Principal Investigator:** Mugole Joel
**Academic Programme:** Bachelor of Science in Computer Science (Year 2)
**Institution:** Makerere University
**Location:** Kampala, Uganda
**Research Domain:** Artificial Intelligence for Healthcare / Digital Health / Clinical Machine Learning
**Project Type:** Research + Deployable Clinical AI System
**Target Environment:** Ugandan Public Healthcare Facilities

---

# Abstract

Uganda is currently facing a dangerous dual disease burden in which communicable diseases such as HIV/AIDS coexist with rapidly increasing non-communicable diseases, particularly hypertension and cardiovascular complications. Patients receiving antiretroviral therapy (ART) are now surviving longer and increasingly developing hypertension, stroke risk, and metabolic disorders. However, healthcare facilities in Uganda operate under severe staffing constraints, inconsistent documentation practices, and limited digital infrastructure.

This project proposes the Uganda Integrated Care Intelligence System (UGICIS), a missingness-aware multi-task clinical machine learning system designed specifically for low-resource African healthcare environments. UGICIS uses real Ugandan HIV-hypertension clinical data to simultaneously predict hypertension presence, hypertension severity, urgent treatment need, and clinical referral pathways.

The core scientific contribution of the project is the MissAware-MTL architecture — a Missingness-Aware Multi-Task Learning framework that treats incomplete clinical records not as noise to be discarded, but as informative clinical signals reflecting real-world healthcare constraints.

The system integrates:

* FT-Transformer architecture for tabular healthcare intelligence
* Missingness indicator embeddings
* Multi-task learning
* SHAP explainability
* Monte Carlo Dropout uncertainty estimation
* Clinical pathway routing logic
* Offline-capable deployment design

UGICIS is designed not merely as an academic machine learning experiment, but as the foundation of a clinically deployable decision-support system for Ugandan healthcare workers.

---

# 1. Problem Background

## 1.1 Uganda’s Dual Disease Burden

Uganda has achieved major success in HIV treatment through expanded ART programmes. More than 1.4 million Ugandans are currently on ART, and viral suppression rates exceed 90% in many districts.

However, this success has created a new challenge:

HIV-positive patients are now surviving long enough to develop chronic non-communicable diseases such as:

* Hypertension
* Stroke
* Heart failure
* Cardiovascular disease
* Metabolic complications

Studies conducted by MRC/UVRI and LSHTM Uganda Research Unit indicate that hypertension prevalence among HIV-positive Ugandans on ART ranges between 25% and 40%, compared to approximately 15% in the general adult population.

Despite this elevated risk:

* hypertension screening is inconsistent
* healthcare workers are overloaded
* clinical decisions remain mostly manual
* referral systems are inefficient
* AI systems trained on Western datasets fail under African clinical realities

---

# 2. Core Problems Identified

## Problem 1 — Missing Clinical Data

African healthcare datasets routinely exhibit:

* incomplete measurements
* skipped documentation
* interrupted patient follow-up
* missing laboratory values
* inconsistent data entry

Traditional machine learning systems:

* discard incomplete records
* use naive imputation
* assume data is fully structured

This creates a mismatch between AI assumptions and Ugandan clinical realities.

UGICIS instead treats missingness itself as clinically informative.

---

## Problem 2 — Lack of Intelligent Referral Logic

Healthcare workers often refer patients manually without:

* severity-based routing
* nearest capable facility identification
* local treatment capability analysis
* facility-aware escalation support

This causes:

* unnecessary Mulago referrals
* overloaded tertiary hospitals
* delayed treatment
* transport burden for patients

---

## Problem 3 — Lack of Explainability

A black-box healthcare AI system is unlikely to be trusted clinically.

Healthcare workers need:

* interpretable predictions
* confidence scores
* feature importance explanations
* actionable recommendations

UGICIS integrates explainable AI (XAI) to support transparent clinical reasoning.

---

# 3. Real-World Clinical Scenarios

## Scenario 1 — Missed Hypertension Detection

A 42-year-old HIV-positive patient attends a routine ART clinic visit.

Clinical indicators:

* Blood Pressure: 148/96
* BMI: Overweight
* Alcohol use: Yes
* ART duration: 3 years

The clinic is overcrowded. The nurse misses the hypertension risk.

Three months later the patient is admitted with a stroke.

UGICIS would instead output:

* HIGH RISK
* Stage 2 Hypertension
* Urgent Treatment Needed
* Treat at Current Facility
* Confidence: 88%

---

## Scenario 2 — Incorrect Referral

A patient with elevated blood pressure is referred directly to Mulago despite nearby district-level treatment availability.

UGICIS instead evaluates:

* severity
* nearest capable facility
* facility level
* referral necessity

The system routes the patient to the nearest suitable treatment centre.

---

## Scenario 3 — Missing Data Situation

A clinic’s weighing scale is broken.

BMI and weight are unavailable.

Standard ML systems may fail or produce unreliable outputs.

UGICIS instead:

* detects missingness
* embeds missingness indicators
* reduces prediction confidence appropriately
* still generates clinically useful recommendations

---

# 4. Research Objectives

## Primary Research Aim

To design, implement, validate, and explain a Missingness-Aware Multi-Task Clinical Intelligence System for integrated HIV-hypertension prediction and clinical pathway support using real Ugandan healthcare data.

---

## Specific Objectives

1. Perform comprehensive exploratory data analysis (EDA) on Ugandan HIV-hypertension clinical data.

2. Simulate realistic missingness patterns (MCAR and MNAR).

3. Compare multiple missing-data handling strategies.

4. Build baseline classical machine learning models.

5. Develop the MissAware-MTL architecture.

6. Implement SHAP-based explainability.

7. Evaluate uncertainty estimation via Monte Carlo Dropout.

8. Develop intelligent clinical routing logic.

9. Produce an offline-deployable AI-ready model architecture.

---

# 5. Dataset Information

## Primary Dataset

| Attribute         | Detail                                              |
| ----------------- | --------------------------------------------------- |
| Dataset Name      | Uganda Integrated HIV-Hypertension Clinical Dataset |
| Repository        | Dryad Digital Repository                            |
| DOI               | doi:10.5061/dryad.9p8cz8wqg                         |
| Records           | 2,645 patient records                               |
| Clinical Features | 19 structured variables                             |
| Source            | Ugandan public healthcare facilities                |
| Data Type         | De-identified clinical records                      |
| Ethics            | Covered under existing approvals                    |

---

## Why This Dataset Is Strong

The dataset:

* contains real Ugandan patients
* reflects real Ugandan healthcare workflows
* is peer-reviewed and scientifically credible
* directly aligns with the target deployment environment
* supports clinically meaningful prediction tasks

Unlike generic Kaggle datasets, this dataset has strong ecological validity.

---

# 6. Dataset Features

| Feature        | Description                      |
| -------------- | -------------------------------- |
| age_category   | Patient age group                |
| female         | Gender indicator                 |
| artyr          | Years on ART                     |
| alcohol        | Alcohol use                      |
| smoke          | Smoking status                   |
| overweight     | BMI / overweight status          |
| exercise       | Physical activity level          |
| marital_status | Marital status                   |
| bpmdate6mo     | Blood pressure measured recently |
| htn_now        | Current hypertension status      |
| htn_stage      | Hypertension severity stage      |
| treat          | Urgent treatment required        |
| hc_code        | Health centre identifier         |
| hc4            | Health centre level              |
| bpfinal        | Final blood pressure reading     |

---

# 7. Type of Machine Learning Problem

UGICIS is primarily a:

# Multi-Task Classification Problem

The system predicts multiple clinically related outputs simultaneously.

---

## Prediction Tasks

| Task                  | Output Type                |
| --------------------- | -------------------------- |
| Hypertension Presence | Binary Classification      |
| Hypertension Severity | Multi-Class Classification |
| Urgent Treatment Need | Binary Classification      |
| Clinical Routing      | Multi-Class Classification |

---

# 8. Missing Data Strategy

## Why Missingness Matters

In African healthcare environments, missing values often reflect:

* overloaded clinics
* equipment shortages
* interrupted care
* staff shortages
* irregular patient follow-up

Therefore:

# Missingness itself contains clinical information.

---

# 9. Missingness Simulation

Because the dataset contains limited natural missingness, realistic missingness patterns will be simulated.

## MCAR — Missing Completely At Random

Random feature removal simulating:

* skipped documentation
* accidental omissions

---

## MNAR — Missing Not At Random

Structured missingness simulating:

* sicker patients
* overloaded facilities
* inconsistent follow-up

Example:

Older patients may disproportionately lack weight measurements.

---

# 10. Missingness Handling Strategies

| Strategy               | Description                       |
| ---------------------- | --------------------------------- |
| Complete Case Analysis | Remove incomplete rows            |
| Mean Imputation        | Replace with averages             |
| MICE                   | Iterative multivariate imputation |
| k-NN Imputation        | Use nearest patients              |
| MIWAE                  | Deep generative imputation        |
| MissAware Embedding    | Learn from missingness directly   |

---

# 11. Machine Learning Pipeline

# Full Clinical AI Pipeline

```text
Raw Ugandan Clinical Dataset
                ↓
Exploratory Data Analysis (EDA)
                ↓
Missingness Analysis
                ↓
Missingness Simulation (MCAR + MNAR)
                ↓
Feature Engineering
                ↓
Categorical Encoding
                ↓
Train/Test Split
                ↓
SMOTE on Training Set Only
                ↓
Baseline Classical Models
(Logistic Regression, RF, XGBoost, CatBoost)
                ↓
MissAware-MTL Deep Learning Model
                ↓
Hyperparameter Optimization
                ↓
Model Evaluation
                ↓
SHAP Explainability
                ↓
Clinical Routing Logic
                ↓
Final Clinical Decision Output
```

---

# 12. Train/Test Splitting Strategy

The dataset will first be divided into:

| Dataset Partition | Percentage |
| ----------------- | ---------- |
| Training Set      | 80%        |
| Testing Set       | 20%        |

Important:

* SMOTE is applied ONLY on training data
* test data remains untouched and real
* stratified splitting preserves class distribution

---

# 13. Baseline Classical Models

Before deep learning, strong classical baselines must be established.

## Baseline Models

| Model               | Purpose                            |
| ------------------- | ---------------------------------- |
| Logistic Regression | Interpretable linear baseline      |
| Decision Tree       | Rule-based clinical logic          |
| Random Forest       | Strong ensemble baseline           |
| XGBoost             | High-performance gradient boosting |
| LightGBM            | Efficient boosting                 |
| CatBoost            | Optimized for categorical features |

---

# 14. Why XGBoost Is Important

For small structured datasets, gradient boosting models often outperform deep learning.

Therefore:

MissAware-MTL must demonstrate meaningful improvements over strong boosting baselines.

This strengthens the scientific credibility of the research.

---

# 15. Proposed Deep Learning Architecture

# MissAware-MTL Architecture

```text
PATIENT CLINICAL RECORD
        ↓
MISSINGNESS INDICATOR EMBEDDING
        ↓
FT-TRANSFORMER SHARED ENCODER
        ↓
 ┌────────┬────────┬────────┬────────┐
 │ HEAD 1 │ HEAD 2 │ HEAD 3 │ HEAD 4 │
 │ HTN    │ STAGE  │ TREAT  │ ROUTE  │
 └────────┴────────┴────────┴────────┘
        ↓
MONTE CARLO DROPOUT
        ↓
UNCERTAINTY ESTIMATION
        ↓
SHAP EXPLAINABILITY
        ↓
FINAL CLINICAL OUTPUT
```

---

# 16. Architecture Components

## Component 1 — Missingness Embedding Layer

Each feature is represented as:

* feature value
* missingness indicator

Example:

| Feature       | Encoded Form |
| ------------- | ------------ |
| Present Value | [X, 0]       |
| Missing Value | [0, 1]       |

This allows the model to learn from patterns of absence.

---

## Component 2 — FT-Transformer Encoder

The FT-Transformer applies attention mechanisms to structured clinical data.

Advantages:

* learns feature interactions
* captures complex clinical relationships
* supports multi-task learning
* scalable to future healthcare tasks

---

## Component 3 — Multi-Task Prediction Heads

| Head   | Prediction            |
| ------ | --------------------- |
| Head 1 | Hypertension Presence |
| Head 2 | Hypertension Severity |
| Head 3 | Urgent Treatment      |
| Head 4 | Clinical Routing      |

---

## Component 4 — Uncertainty Estimation

Monte Carlo Dropout runs predictions multiple times.

The model outputs:

* prediction
* confidence score
* uncertainty level

This is critical for safe healthcare deployment.

---

# 17. Multi-Task Loss Function

The total training loss combines all prediction tasks:

L_total = λ1L_htn + λ2L_stage + λ3L_treat + λ4L_route

Where:

* L_htn = hypertension classification loss
* L_stage = stage prediction loss
* L_treat = treatment urgency loss
* L_route = routing prediction loss

---

# 18. Explainable AI (XAI)

UGICIS integrates:

* SHAP feature importance
* Counterfactual explanations
* Confidence estimation

Example output:

Why is this patient HIGH RISK?

* Overweight status: +34%
* Alcohol use: +21%
* No exercise: +18%
* ART duration: +12%

Confidence: 87%

---

# 19. Evaluation Metrics

| Metric               | Importance       |
| -------------------- | ---------------- |
| Recall (Sensitivity) | Highest Priority |
| F1 Score             | High             |
| ROC-AUC              | High             |
| Precision            | Medium           |
| Calibration Error    | High             |

---

# 20. Validation Strategy

UGICIS uses:

* Stratified Train/Test Split
* 5-Fold Cross Validation
* SMOTE on Training Only
* Fairness Analysis
* Facility-Level Validation

---

# 21. Hyperparameter Optimization

Optimization tools:

* Optuna
* GridSearchCV
* RandomizedSearchCV

Parameters tuned:

* learning rate
* dropout rate
* attention heads
* hidden dimensions
* tree depth
* estimators

---

# 22. Technology Stack

| Component            | Technology                  |
| -------------------- | --------------------------- |
| Programming Language | Python 3.10+                |
| Deep Learning        | PyTorch                     |
| Classical ML         | Scikit-learn                |
| Boosting Models      | XGBoost, LightGBM, CatBoost |
| Explainability       | SHAP, DICE-ML               |
| Optimization         | Optuna                      |
| Experiment Tracking  | MLflow                      |
| Data Processing      | Pandas, NumPy               |
| Visualization        | Matplotlib, Seaborn         |
| Deployment           | FastAPI                     |
| Frontend             | React.js + Tailwind CSS     |
| Containerization     | Docker                      |

---

# 23. Repository Structure

```text
uganda-integrated-care-intelligence-system/
├── data/
├── notebooks/
├── preprocessing/
├── models/
├── training/
├── evaluation/
├── explainability/
├── routing/
├── deployment/
├── configs/
├── requirements.txt
├── README.md
└── train.py
```

---

# 24. Expected Deliverables

| Deliverable                | Description                   |
| -------------------------- | ----------------------------- |
| Jupyter Notebooks          | Full reproducible ML pipeline |
| Final Research Report      | Complete academic write-up    |
| Baseline Comparison Tables | Model benchmarking            |
| SHAP Visualizations        | Explainable AI outputs        |
| Clinical Routing Prototype | Intelligent referral logic    |
| GitHub Repository          | Open-source implementation    |
| Deployment Foundation      | ONNX-ready export             |

---

# 25. Future Work — Phase 2

Future extensions include:

* Offline Android dashboard
* Progressive Web App (PWA)
* Facility GPS integration
* DHIS2 integration
* Federated learning
* National health intelligence dashboards
* Real-time referral intelligence

---

# 26. Ethical Considerations

* No personally identifiable information used
* Dataset fully de-identified
* AI supports — not replaces — clinicians
* Fairness analysis included
* Uncertainty estimation provided
* Open-source reproducibility prioritized

---

# 27. Why This Project Matters

UGICIS is not simply another machine learning experiment.

It is:

* locally relevant
* clinically grounded
* technically rigorous
* explainable
* scalable
* deployable

The project addresses one of the most important gaps in African healthcare AI:

# building intelligent systems designed for real African clinical environments rather than importing assumptions from Western healthcare systems.

---

# 28. Author

## Mugole Joel

Bachelor of Science in Computer Science
Makerere University
Kampala, Uganda

GitHub: Robotics1025

---

# 29. License

This project is intended for academic research and educational purposes.

---

# 30. Final Vision

UGICIS represents the beginning of a broader African Healthcare Foundation Model capable of:

* integrated disease intelligence
* multi-condition clinical prediction
* low-resource healthcare deployment
* explainable healthcare AI
* intelligent healthcare routing
* population-level African health analytics

The long-term vision is to build AI systems designed not only for hospitals with unlimited infrastructure, but for the real healthcare environments where most Africans receive care every day.
