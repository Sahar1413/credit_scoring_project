# Credit Scoring Project

This project was developed during a 1-month internship at CodeAlpha.  
It uses machine learning to **predict the creditworthiness** of borrowers using the **German Credit dataset**.

🛠️ Features

-Data preprocessing & feature engineering
- Model training:Random Forest with hyperparameter tuning
- Evaluation metrics:** Accuracy, F1-score, Confusion Matrix, ROC curve, ROC-AUC
- Feature importance analysis with visualization
- Reports:CSV and PNG figures for interpretability

 📊 Results

- ROC-AUC: 0.804
- Top features impacting credit risk: 
  - Credit_Amount  
  - Status_Checking_A14 (current account status)  
  - Duration  
  - Age  

- Visualizations generated:
  - Feature importance  
  - Confusion matrix  
  - ROC curve

 💻 Installation

1. Clone the repository:
git clone https://github.com/Sahar1413/credit_scoring_project.git
cd credit_scoring_project
2. Install dependencies:
pip install -r requirements.txt
🚀 Usage

1. Train the model:
python3 src/train_model.py
2. Evaluate the model:
python3 src/evaluate_model.py
📁 Project Structure

credit_scoring_project/
├── src/                 # Scripts (training, evaluation)
├── models/              # Trained model file
├── reports/             # Figures and CSVs
├── preprocessing.py     # Data preparation functions
└── README.md
 🔗 GitHub

Project repository: [https://github.com/Sahar1413/credit_scoring_project](https://github.com/Sahar1413/credit_scoring_project)
⚡ Notes

* The project demonstrates how machine learning can support financial decision-making.
* Written in Python using scikit-learn, pandas, matplotlib.
