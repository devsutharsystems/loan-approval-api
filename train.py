import pickle
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from preprocessing_utils import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    encode_features,
    fill_missing_values,
    scale_features,
)


def load_data(path: str = "train.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df = fill_missing_values(df)
    return df


def build_encoders() -> dict:
    """One LabelEncoder per categorical column, fit once on training data."""
    return {col: LabelEncoder() for col in CATEGORICAL_COLUMNS}


def train_and_compare_models(X_train, X_test, y_train, y_test) -> dict:
    """Train all 5 classifiers and return their fitted instances + accuracy scores."""
    models = {
        "Random Forest": RandomForestClassifier(random_state=0),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=0),
        "KNN": KNeighborsClassifier(),
        "SVC": SVC(probability=True, random_state=0),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        results[name] = {"model": model, "accuracy": accuracy}
        print(f"{name}: {accuracy:.4f} accuracy")

    return results


def plot_model_comparison(results: dict, save_path: str = "model_comparison.png"):
    names = list(results.keys())
    accuracies = [results[n]["accuracy"] * 100 for n in names]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, accuracies, color="#2563eb")
    plt.ylabel("Accuracy (%)")
    plt.title("Model Comparison - Loan Approval Prediction")
    plt.ylim(0, 100)
    plt.xticks(rotation=20)

    for bar, acc in zip(bars, accuracies):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{acc:.1f}%",
            ha="center",
        )

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved comparison chart to {save_path}")


def main():
    # 1. Load and clean data
    data = load_data("train.csv")

    # 2. Train/test split (done BEFORE encoding/scaling to avoid data leakage)
    X_raw = data[FEATURE_COLUMNS]
    y_raw = data["Loan_Status"]

    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X_raw, y_raw, test_size=0.25, random_state=0
    )

    # 3. Fit encoders on TRAINING data only, then apply consistently to both splits
    encoders = build_encoders()
    X_train = encode_features(X_train_raw, encoders, fit=True)
    X_test = encode_features(X_test_raw, encoders, fit=False)

    label_encoder_y = LabelEncoder()
    y_train = label_encoder_y.fit_transform(y_train_raw)
    y_test = label_encoder_y.transform(y_test_raw)

    # 4. Fit scaler on TRAINING data only, then apply consistently to both splits
    scaler = StandardScaler()
    X_train = scale_features(X_train, scaler, fit=True)
    X_test = scale_features(X_test, scaler, fit=False)

    # 5. Train and compare all 5 classifiers
    results = train_and_compare_models(X_train, X_test, y_train, y_test)
    plot_model_comparison(results)

    # 6. Select the best model by test accuracy
    best_name = max(results, key=lambda n: results[n]["accuracy"])
    best_model = results[best_name]["model"]
    print(f"\nBest model: {best_name} ({results[best_name]['accuracy']:.4f} accuracy)")

    # 7. Save the best model, encoders, scaler, and label encoder for the target
    with open("model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    with open("encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)

    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    with open("label_encoder_y.pkl", "wb") as f:
        pickle.dump(label_encoder_y, f)

    print("Saved model.pkl, encoders.pkl, scaler.pkl, label_encoder_y.pkl")


if __name__ == "__main__":
    main()
