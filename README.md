# Credit Card Fraud Detection Using Machine Learning

B.Tech CSE (AI/ML) Mini/Major Project

## Project objective
Classify a credit-card transaction as Legitimate or Fraudulent using machine learning.

## Models
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Isolation Forest (anomaly detection)

## Important
For the final academic project, use a real public credit-card fraud dataset (for example, the commonly used Credit Card Fraud Detection dataset) instead of the included demo data. Put the downloaded CSV in `data/creditcard.csv`.

The included `generate_demo_data.py` creates a synthetic dataset only so that the project can be tested immediately.

## Run
```bash
pip install -r requirements.txt
python generate_demo_data.py
python train_model.py
streamlit run app.py
```

After training, the model files are stored in `models/`.

## Dataset format
The code expects a CSV containing a target column named `Class`:
- 0 = legitimate
- 1 = fraud

It automatically handles numeric features and scales them before training.

## Team division
Member 1: data preprocessing + EDA
Member 2: model training + evaluation
Member 3: Streamlit application + integration
