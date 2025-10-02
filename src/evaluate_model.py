"""
📌 Credit Scoring Model Evaluation (No PDF)
"""
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from preprocessing import prepare_data

# ============================
# Paths
# ============================
model_path = os.path.join("..", "models", "credit_scoring_model.pkl")
figures_dir = os.path.join("..", "reports", "figures")
os.makedirs(figures_dir, exist_ok=True)

# ============================
# Load data and model
# ============================
_, X_test, _, y_test, _ = prepare_data()
model = joblib.load(model_path)

# ============================
# Feature Importance
# ============================
def get_feature_names(preprocessor):
    feature_names = []
    for name, transformer, cols in preprocessor.transformers_:
        if hasattr(transformer, 'named_steps'):
            for step_name, step_transformer in transformer.named_steps.items():
                if hasattr(step_transformer, 'get_feature_names_out'):
                    feature_names.extend(step_transformer.get_feature_names_out(cols))
                    break
            else:
                feature_names.extend(cols)
        elif hasattr(transformer, 'get_feature_names_out'):
            feature_names.extend(transformer.get_feature_names_out(cols))
        else:
            feature_names.extend(cols)
    return list(feature_names)

def plot_feature_importance(clf, feature_names, top_n=20):
    if hasattr(clf.named_steps['classifier'], "feature_importances_"):
        importances = clf.named_steps['classifier'].feature_importances_
    elif hasattr(clf.named_steps['classifier'], "coef_"):
        importances = abs(clf.named_steps['classifier'].coef_).ravel()
    else:
        raise ValueError("Model does not support feature importance or coefficients.")

    indices = importances.argsort()[-top_n:]
    plt.figure(figsize=(10,6))
    plt.barh(range(len(indices)), importances[indices][::-1], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices][::-1])
    plt.title("Top Features for Credit Scoring")
    plt.xlabel("Feature Importance")
    plt.tight_layout()
    
    path = os.path.join(figures_dir, "feature_importance.png")
    plt.savefig(path)
    plt.show()
    return path, importances, indices

def export_feature_importance(clf, feature_names):
    if hasattr(clf.named_steps['classifier'], "feature_importances_"):
        importances = clf.named_steps['classifier'].feature_importances_
    elif hasattr(clf.named_steps['classifier'], "coef_"):
        importances = abs(clf.named_steps['classifier'].coef_).ravel()
    else:
        raise ValueError("Model does not support feature importance or coefficients.")

    df = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values(by="Importance", ascending=False)
    csv_path = os.path.join("..", "reports", "feature_importance.csv")
    df.to_csv(csv_path, index=False)
    return csv_path, df

preprocessor = model.named_steps['preprocessor']
feature_names = get_feature_names(preprocessor)
feat_img, importances, indices = plot_feature_importance(model, feature_names)
feat_csv, df_features = export_feature_importance(model, feature_names)

# ============================
# Predictions & metrics
# ============================
y_pred = model.predict(X_test)
y_score = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else y_pred

# Classification report
print("📊 Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Solvent','Non-solvent'], yticklabels=['Solvent','Non-solvent'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
cm_path = os.path.join(figures_dir, "confusion_matrix.png")
plt.savefig(cm_path)
plt.show()

# ROC curve plot
fpr, tpr, _ = roc_curve(label_binarize(y_test, classes=[0,1]).ravel(), y_score)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
roc_path = os.path.join(figures_dir, "roc_curve.png")
plt.savefig(roc_path)
plt.show()

print(f"✅ Feature importance CSV: {feat_csv}")
print(f"✅ Feature importance figure: {feat_img}")
print(f"✅ Confusion matrix figure: {cm_path}")
print(f"✅ ROC curve figure: {roc_path}")
print(f"📈 ROC-AUC: {roc_auc:.3f}")
# ============================
# Top 5 feature interpretation (console output)
# ============================
top_features = df_features.head(5)
print("\n🔹 Top 5 Feature Interpretation:")
for i, row in top_features.iterrows():
    if "Credit_Amount" in row["Feature"]:
        print(f"💰 {row['Feature']} → Higher credit amount may increase default risk.")
    elif "Duration" in row["Feature"]:
        print(f"⏳ {row['Feature']} → Longer repayment duration may increase risk.")
    elif "Age" in row["Feature"]:
        print(f"👤 {row['Feature']} → Borrower's age can influence financial stability.")
    elif "Status_Checking" in row["Feature"]:
        print(f"🏦 {row['Feature']} → Current account status indicates solvency.")
    else:
        print(f"🔹 {row['Feature']} → Important for credit scoring.")
