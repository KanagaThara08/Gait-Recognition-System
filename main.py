import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from utils import load_dataset
from sklearn.preprocessing import LabelEncoder

def mean_reciprocal_rank(y_true, y_pred_prob):
    ranks = []
    for true_label, probs in zip(y_true, y_pred_prob):
        sorted_indices = np.argsort(probs)[::-1]
        try:
            rank = np.where(sorted_indices == true_label)[0][0] + 1
            ranks.append(1.0 / rank)
        except:
            ranks.append(0)
    return np.mean(ranks)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <dataset_path>")
        sys.exit(1)

    dataset_path = sys.argv[1]

    print("[INFO] Loading dataset...")
    X, y = load_dataset(dataset_path)

    if len(X) == 0:
        print("[ERROR] No data found. Exiting.")
        sys.exit(1)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print(f"[INFO] Total Samples: {len(X)} | Total Subjects: {len(set(y))}")
    print("[INFO] Training Random Forest Classifier...\n")

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    mrr = mean_reciprocal_rank(y_test, y_pred_prob)

    print("\n========== Gait Recognition Results ==========")
    print(f"[ACCURACY]: {acc * 100:.2f}%")
    print(f"[MRR]: {mrr:.4f} (Mean Reciprocal Rank)")

    print("\n[DETAILED CLASSIFICATION REPORT]:\n")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
