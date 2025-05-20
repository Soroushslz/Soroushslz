# MS

# 🧠 Diagnosis of Multiple Sclerosis Using Hybrid CNNs and Multimodal Data

This repository presents a deep learning pipeline for diagnosing **Multiple Sclerosis (MS)** by integrating **3D Brain MRI scans** and **clinical metadata** using **hybrid Convolutional Neural Networks (CNNs)** and a **multimodal fusion architecture**. Our approach demonstrates robust performance by combining the strengths of image-based and tabular data models.

![MRI Fusion Architecture](https://user-images.githubusercontent.com/placeholder/fusion-architecture.png) <!-- Optional visual banner if available -->

---

## 📌 Abstract

Multiple Sclerosis (MS) is a chronic autoimmune and neurodegenerative disease affecting the central nervous system. Accurate diagnosis is critical for early intervention and improved patient outcomes. In this project:

* We preprocess **3D MRI** data and extract features alongside structured **clinical attributes**.
* Separate models are trained for **MRI images** using a 3D CNN and **clinical data** using a dense neural network.
* A **fusion model** combines outputs from both modalities to improve classification accuracy.
* **SHAP (SHapley Additive exPlanations)** is used for interpretability and analysis of feature contributions.
* Performance metrics such as **Accuracy**, **F1-score**, **Specificity**, and **ROC-AUC** are used for evaluation.

---

## 🔍 Key Highlights

* **Hybrid Deep Learning Pipeline**: Fusion of CNN-based MRI and clinical models.
* **3D CNN Architecture**: Tailored for volumetric MRI data.
* **Model Performance**:

  * Accuracy: **95.22%**
  * F1 Score: **98.83%**
  * Recall: **100%**
* **Multimodal Integration**: Seamless handling of heterogeneous clinical and imaging data.
* **Interpretability with SHAP**: Visualization of feature importance to support decision-making.
* **Robust Evaluation**: Validated on held-out test data using comprehensive metrics.

---

## 📁 Project Structure

```
📦 MS-Diagnosis-Hybrid-CNN
 ┣ 📊 patient info.xlsx
 ┣ 🧠 preprocessed_mri_data.npy
 ┣ 📜 main.py
 ┣ 📄 README.md
 ┗ 📁 Brain MRI Dataset of MS/
```

---

## 🧬 Data Description

### 1. **Clinical Dataset** (`patient info.xlsx`)

Includes demographic and clinical features such as:

* Age
* Sex
* Disease duration
* Clinical disability scores

### 2. **MRI Dataset** (`preprocessed_mri_data.npy`)

* Preprocessed 3D volumetric brain MRIs of MS patients.
* Each scan is resized to shape `(64, 64, 64, 2)` to maintain spatial consistency.

---

## 🧠 Model Architecture

### MRI Model

* 3D CNN layers
* MaxPooling & Dropout regularization
* Final dense layer with sigmoid activation

### Clinical Model

* Fully connected dense network
* Dropout layers to prevent overfitting

### Fusion Model

* Concatenation of CNN and clinical model outputs
* Final sigmoid layer for binary classification

---

## 🧪 Evaluation Metrics

| Metric      | Score    |
| ----------- | -------- |
| Accuracy    | 95.22%   |
| F1 Score    | 98.83%   |
| Recall      | 100%     |
| Specificity | \~92.00% |
| ROC-AUC     | High     |

---

## 📊 Explainability with SHAP

We used **SHAP** to explain model decisions:

* Clinical Model:

  * Age and EDSS score had minor influence.
* MRI Model:

  * MRI voxel patterns were primary contributors.

This suggests the **MRI modality holds stronger predictive power** in MS diagnosis.

---

## 🚀 How to Run

### Requirements

```bash
pip install tensorflow pandas numpy scikit-learn shap openpyxl
```

### Run the Model

```bash
python main.py
```

> Make sure the dataset files (`.xlsx` and `.npy`) are in the same directory as `main.py`.

---

## 📈 Sample Output

```
Train Accuracy: 95.22%
Test Accuracy: 94.85%
F1-score: 98.83
Specificity: 92.00
```

SHAP visualizations will be displayed for both clinical and MRI feature importances.

---

## 📚 Citation

If you use this repository for your research, please cite:

```
@article{soltanizadeh2025msdiagnosis,
  title={Diagnosis of Multiple Sclerosis Using Hybrid CNNs and Multimodal Data},
  author={Soltanizadeh, Soroush},
  year={2025},
  journal={GitHub Repository}
}
```

---

## 👤 Author

**Soroush Soltanizadeh**

* 🔗 [LinkedIn](https://www.linkedin.com/in/soroush-soltanizadeh-1136892b6/)
* 📚 [Google Scholar](https://scholar.google.com/citations?user=ARKNJYwAAAAJ&hl=en)

---

## 📌 Future Work

* Integration of **BiLSTM** for temporal modeling of MRI slices.
* Deployment as a **web-based diagnostic tool**.
* Further evaluation with **larger, real-world datasets**.

---

## 🛡 License

This project is open-source and licensed under the [MIT License](LICENSE).
