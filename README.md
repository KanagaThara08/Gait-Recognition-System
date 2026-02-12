# Gait Recognition System

## **Project Description**

The Gait Recognition System is a biometric identification project that recognizes individuals based on their walking patterns using image processing and machine learning techniques. The system extracts gait silhouettes, generates features using Histogram of Oriented Gradients (HOG), and classifies subjects using trained models.

---

## **Objective**

* Identify individuals using gait patterns
* Extract silhouette features from gait images
* Train machine learning models for classification
* Evaluate recognition accuracy

---

## **Dataset**

CASIA-B Gait Dataset
(Full dataset is not uploaded due to size constraints. A sample dataset is provided.)

---

## **Methodology**

* Image preprocessing (grayscale conversion and resizing)
* Silhouette feature extraction using HOG
* Label encoding
* Train-test split (80:20)
* Model training:

  * Random Forest
  * Multi-Layer Perceptron (MLP)
* Performance evaluation using Accuracy, ROC Curve, and Precision-Recall Curve

---

## **Results**

| Model         | Accuracy |
| ------------- | -------- |
| Random Forest | 97.74%   |
| MLP           | 80.78%   |

Best performance was achieved using the Random Forest Classifier.

---

## **How to Run**

```
pip install -r requirements.txt
python main.py
```

---

## **Technologies Used**

* Python
* OpenCV
* Scikit-learn
* NumPy
* Matplotlib

---

## **Authors**

Kanaga Thara S

VIT Vellore – SCORE
