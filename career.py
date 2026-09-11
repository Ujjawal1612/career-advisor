import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_PATH = "career_pred.csv"
MODEL_PATH = "career_model.joblib"

FEATURES = [
    "sslc", "hsc", "cgpa", "school_type", "no_of_miniprojects", "no_of_projects",
    "coresub_skill", "aptitude_skill", "problemsolving_skill", "programming_skill",
    "abstractthink_skill", "design_skill", "first_computer", "first_program",
    "lab_programs", "ds_coding", "technology_used", "sympos_attend", "sympos_won",
    "extracurricular", "learning_style", "college_bench", "clg_teachers_know",
    "college_performence", "college_skills"
]


def load_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=["ROLE"]).copy()
    df["ROLE"] = df["ROLE"].astype(str).str.strip()
    df = df[df["ROLE"] != ""]

    for column in FEATURES:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df.dropna(subset=FEATURES)


def train():
    df = load_data()
    X = df[FEATURES]
    encoder = LabelEncoder()
    y = encoder.fit_transform(df["ROLE"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Rows used: {len(df)}")
    print(f"Number of careers: {len(encoder.classes_)}")
    print(f"Test accuracy: {accuracy:.2%}")
    print("\nClassification report:\n")
    print(classification_report(y_test, predictions, target_names=encoder.classes_, zero_division=0))

    artifact = {
        "model": model,
        "label_encoder": encoder,
        "features": FEATURES,
        "accuracy": float(accuracy),
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")
    return artifact


if __name__ == "__main__":
    train()
