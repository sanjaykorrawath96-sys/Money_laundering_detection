# 🛡️ Intelligent Anti-Money Laundering Detection System (ANN)

A Streamlit web app that classifies a bank transaction as **legitimate** or **suspicious** using an Artificial Neural Network (ANN) built with TensorFlow/Keras.

## Business Problem

Rule-based AML systems generate many false alerts and miss sophisticated laundering patterns. This project uses historical transaction data to train a binary classifier that supports transaction monitoring and regulatory compliance.

## Dataset

- **Source:** [SAML-D: Synthetic Transaction Monitoring Dataset (Kaggle)](https://www.kaggle.com/datasets/berkanoztas/synthetic-transaction-monitoring-dataset-aml)
- **Size:** 9,504,852 transactions, 12 columns
- **Target:** `Is_laundering` (0 = legitimate, 1 = suspicious)
- **Class imbalance:** about 0.1% of transactions are suspicious
- **Training sample:** 100,000 randomly sampled rows (113 suspicious)

### Features used

| Feature | Type |
|---|---|
| Amount | Numeric |
| Payment_currency | Categorical |
| Received_currency | Categorical |
| Sender_bank_location | Categorical |
| Receiver_bank_location | Categorical |
| Payment_type | Categorical |

Dropped columns: `Time`, `Date`, `Sender_account`, `Receiver_account`, `Laundering_type`.

## Methodology

1. **EDA:** univariate and bivariate analysis, class distribution, payment-type and currency patterns.
2. **Preprocessing:**
   - One-Hot Encoding for categorical columns
   - Standard Scaling for `Amount`
   - Combined with `ColumnTransformer` (70 features after encoding)
3. **Imbalance handling:** SMOTE oversampling (balanced to 50/50).
4. **Model:** Sequential ANN
   - 3 × Dense(22, ReLU) with Dropout (0.19)
   - Output: Dense(1, Sigmoid)
   - Optimizer: RMSprop (learning rate ≈ 0.0376)
   - Loss: binary cross-entropy, 30 epochs
5. **Hyperparameter tuning:** Optuna (30 trials, minimizing validation loss).
6. **Deployment:** Streamlit app for real-time single-transaction scoring.

## Project Structure

```
├── app.py                 # Streamlit application
├── AML.keras              # Trained ANN model
├── preprocessor           # Fitted ColumnTransformer (joblib)
├── requirements.txt
├── README.md
└── Intelligent_Anti-Money_Laundering_Detection_System_Using_Artificial_Neural_Networks.ipynb
```

## Installation

```bash
git clone <your-repo-url>
cd <your-repo-folder>

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### requirements.txt

```
streamlit
tensorflow
scikit-learn
imbalanced-learn
pandas
numpy
joblib
```

> Use the same scikit-learn and TensorFlow versions that were used to train the model, otherwise loading `preprocessor` and `AML.keras` may fail.

## Run the App

```bash
streamlit run app.py
```

Then open http://localhost:8501.

## How to Use

1. Enter the transaction amount, payment type, currencies, and sender and receiver bank locations.
2. Click **Analyze Transaction**.
3. The app shows the raw model score, the prior-corrected real-world risk, and a verdict (**Suspicious** or **Legitimate**).
4. Use the sidebar slider to set the flag threshold (default 1%).

### Why prior correction?

The model was trained on SMOTE-balanced data (50% suspicious), but real-world laundering is about 0.1%. Raw scores are therefore inflated and flag too many transactions. The app converts the raw score into an estimated real-world probability before applying the threshold.

## Regenerating the model files

Run the notebook to the end. It saves:

```python
model.save('AML.keras')
joblib.dump(preprocess, 'preprocessor')
```

Copy both files next to `app.py`.

## Limitations

- Accuracy on the SMOTE-balanced data is about 52%, so the predictions are not reliable yet.
- No held-out test set was used. Evaluation was on the resampled training data.
- Account-level and time-based behaviour (frequency, velocity, fan-in/fan-out) is not used.
- The data is synthetic and may not reflect real banking patterns.
- Accuracy is a poor metric for this problem. Recall, precision and PR-AUC are more appropriate.
- This is a decision-support tool and should not be the sole basis for compliance decisions.

## Future Improvements

- Split into train and test sets **before** applying SMOTE
- Add engineered features from sender and receiver account history
- Tune the model on recall or PR-AUC
- Try class weights, focal loss, or gradient-boosting baselines
- Add SHAP-based explanations
- Add model monitoring and threshold calibration

## Tech Stack

Python · TensorFlow/Keras · scikit-learn · imbalanced-learn · Optuna · Pandas · Streamlit

