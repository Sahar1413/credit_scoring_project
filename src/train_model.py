# src/train_model.py

import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from preprocessing import prepare_data

def train_model():
    # 1. Load data
    X_train, X_test, y_train, y_test, preprocessor = prepare_data()

    # 2. Define pipeline (preprocessing + model)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(class_weight="balanced", random_state=42))
    ])

    # 3. Hyperparameter grid (note: keys use pipeline step name)
    param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [None, 10, 20],
        "classifier__min_samples_split": [2, 5],
        "classifier__min_samples_leaf": [1, 2]
    }

    # 4. GridSearchCV
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=2
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    # 5. Save best model
    model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(best_model, os.path.join(model_dir, "credit_scoring_model.pkl"))

    print("✅ Best model trained and saved!")
    print("Best parameters:", grid_search.best_params_)

if __name__ == "__main__":
    train_model()
