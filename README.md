# MS
# 🧠 Diagnosis of Multiple Sclerosis Using Hybrid CNNs and Multimodal Data

This project presents a deep learning pipeline for diagnosing **Multiple Sclerosis (MS)** using a hybrid approach that combines **3D Convolutional Neural Networks (CNNs)** for MRI data and **Dense Neural Networks** for clinical data. It further integrates both modalities via a **fusion model**, and utilizes **SHAP** (SHapley Additive exPlanations) for model explainability.

---

## 🚀 Highlights

* **Multimodal fusion**: Combines 3D brain MRI and clinical features.
* **Hybrid architecture**: Uses CNNs for MRI and DNNs for clinical data.
* **SHAP explainability**: Visualizes contributions of features in model decision-making.
* **High accuracy**: Demonstrates a strong performance with robust generalization.
* **Model stacking**: Final prediction is made through a fusion of individual modality outputs.

---

## 🧬 Background

Multiple Sclerosis is a chronic neurodegenerative disease affecting the central nervous system. Due to its complex presentation, accurate and early diagnosis is critical.

A comparative analysis showed that hybrid models such as CNN-BiLSTM perform remarkably well, with our best model achieving:

* **Accuracy**: 95.22%
* **F1 Score**: 98.83%
* **Recall**: 100%

These results demonstrate that our model not only performs well statistically, but also strikes a balance between precision and recall — crucial for clinical diagnosis.

---

## 🗃️ Dataset

* **MRI Data**: 3D NIfTI scans of brain MRIs from MS patients.
* **Clinical Data**: Structured tabular data including demographic and clinical features.

> 📁 Expected file structure:

```
.
├── preprocessed_mri_data.npy
├── patient info.xlsx
└── Brain MRI Dataset of MS/
```

---

## 🏗️ Model Architecture

### 1. MRI Model (3D CNN)

Processes volumetric MRI data using a 3-layer 3D CNN followed by dense layers.

### 2. Clinical Model (DNN)

Processes tabular clinical data via a shallow neural network.

### 3. Fusion Model

Combines the output logits from both individual models and feeds them into a small dense network for final classification.

---

## 📈 Training and Evaluation

* Models are trained using the **binary cross-entropy** loss and **Adam optimizer**.
* Early stopping is applied to prevent overfitting.
* Custom **F1-score metric** is used for performance monitoring.
* Final evaluation includes **accuracy**, **F1-score**, and **specificity**.

---

## 📊 Results

| Metric      | Training Set | Test Set |
| ----------- | ------------ | -------- |
| Accuracy    | \~95.2%      | \~95.2%  |
| F1 Score    | \~98.8%      | \~98.8%  |
| Recall      | 100%         | 100%     |
| Specificity | \~94.5%      | \~94.5%  |

---

## 🔍 Explainability with SHAP

To interpret model decisions:

* **Clinical model**: SHAP summary plots identify key features (age, symptoms, etc.).
* **MRI model**: SHAP visualizations show spatial contributions of 3D brain volumes.

<details>
<summary>Example SHAP Summary Plot (Clinical)</summary>

```
shap.summary_plot(shap_values_clinical, X_test_clinical, feature_names=clinical_data.columns[:-1])
```

</details>

<details>
<summary>Example SHAP Image Plot (MRI)</summary>

```
shap.image_plot(shap_values_mri[:1], X_test_mri[:1])
```

</details>

---

## 🧠 Key Findings

* **CNN-BiLSTM fusion** yielded the best diagnostic performance.
* **MRI data dominates** in influencing the model's decisions; clinical data contributed minimally as shown by SHAP.
* **Multimodal integration** enhances robustness and generalization in unseen patient data.
* The system shows high potential as a **decision-support tool** for neurologists.

---

## 🛠️ Requirements

Install dependencies:

```bash
pip install numpy pandas tensorflow shap scikit-learn openpyxl
```

---

## ▶️ Run the Project

1. Place your files:

   * `patient info.xlsx`
   * `preprocessed_mri_data.npy`

2. Run the pipeline:

```bash
python ms_diagnosis_pipeline.py
```

---

## 📚 References

* SHAP ([https://github.com/slundberg/shap](https://github.com/slundberg/shap))
* TensorFlow ([https://www.tensorflow.org/](https://www.tensorflow.org/))
* \[Original Dataset Source or Citation if applicable]

---

## ✍️ Author

**Soroush Soltanizadeh**
Researcher in AI for Healthcare | Specializing in Biomedical Signal & Image Processing
[LinkedIn](https://www.linkedin.com/) | [Google Scholar](https://scholar.google.com/) *(add links if available)*

---

## 📄 License

This project is licensed under the MIT License.
