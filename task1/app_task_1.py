from flask import Flask, request, jsonify
import joblib  # For loading the ML model
import pandas as pd
import spacy
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from flask_cors import CORS
from xgboost import XGBClassifier,XGBRegressor
import xgboost

sdgsht = {0:"0 to 100",1:"100 to 1000",2:"1000 to 5000",3:"5000 to 10000",4:"10000 to 100000",5:"Above 100000"}

app = Flask(__name__)
CORS(app)  # This will allow CORS for all routes by default

# Load models and tokenizer
nlp = spacy.load("en_core_web_md")
# Load models and tokenizer
classifier = joblib.load('xgboost_model (2).joblib')
y_reg1 = joblib.load("xgb_regressor_model_cat1 (1).joblib")
y_reg2 = joblib.load("xgb_regressor_model_cat2 (1).joblib")
y_reg3 = joblib.load("xgb_regressor_model_cat3 (1).joblib")
y_reg4 = joblib.load("xgb_regressor_model_cat4 (1).joblib")
y_reg5 = joblib.load("xgb_regressor_model_cat5 (1).joblib")
y_reg6 = joblib.load("xgb_regressor_model_cat6 (1).joblib")

print("Models loaded successfully.")
print(f"y_reg1 fitted: {hasattr(y_reg1, 'get_booster')}")
print(f"y_reg2 fitted: {hasattr(y_reg2, 'get_booster')}")
print(f"y_reg3 fitted: {hasattr(y_reg3, 'get_booster')}")
print(f"y_reg4 fitted: {hasattr(y_reg4, 'get_booster')}")
print(f"y_reg5 fitted: {hasattr(y_reg5, 'get_booster')}")
print(f"y_reg6 fitted: {hasattr(y_reg6, 'get_booster')}")

tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False)
bertweet = AutoModel.from_pretrained("vinai/bertweet-base")

device = "cuda" if torch.cuda.is_available() else "cpu"
bertweet.to(device)

def get_bertweet_embeddings(sentences, model, tokenizer, device):
    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

@app.route('/predict-likes', methods=['POST'])
def predict_likes():
    # Get data from the request
    username = request.form['username']
    content = request.form['content']
    company = request.form['company']
    datetime = request.form['datetime']
    
    # Prepare the data for prediction
    input_data = pd.DataFrame({
        'username': [username],
        'content': [content],
        'company': [company],
        'datetime': [datetime],
    })

    # Process the text using spaCy
    docs_content = list(nlp.pipe(input_data['content'], batch_size=1))
    docs_username = list(nlp.pipe(input_data['username'], batch_size=1))
    docs_company = list(nlp.pipe(input_data['company'], batch_size=1))
    docs_date = list(nlp.pipe(input_data['datetime'], batch_size=1))

    # Create vectors
    x_content = np.array([docs_content[0].vector], dtype=np.float32)
    x_username = np.array([docs_username[0].vector], dtype=np.float32)
    x_company = np.array([docs_company[0].vector], dtype=np.float32)
    x_date = np.array([docs_date[0].vector], dtype=np.float32)

    # Concatenate all features
    x = pd.concat([pd.DataFrame(x_content), pd.DataFrame(x_username), pd.DataFrame(x_company), pd.DataFrame(x_date)], axis=1)
    x = np.array(x).astype(np.float32)

    # Predict category
    y_cat = classifier.predict(x)
    print(y_cat)
    tweet = f"{content} {datetime} {username} {company}"

    # Get BERTweet embeddings
    x_tweet = [tweet]
    X_reg = get_bertweet_embeddings(x_tweet, bertweet, tokenizer, device)
    print(f"X_reg shape: {X_reg.shape}")
    # Predict likes based on the category
    if y_cat[0] == 0:  # Ensure you are accessing the first element
        likes = y_reg1.predict(X_reg)
    elif y_cat[0] == 1:
        likes = y_reg2.predict(X_reg)
    elif y_cat[0] == 2:
        likes = y_reg3.predict(X_reg)
    elif y_cat[0] == 3:
        likes = y_reg4.predict(X_reg)
    elif y_cat[0] == 4:
        likes = y_reg5.predict(X_reg)
    elif y_cat[0] == 5:
        likes = y_reg6.predict(X_reg)
    else:
        return jsonify({'error': 'Invalid category'})
    print(y_cat, likes)
    return jsonify({'likes': str(likes[0]),'cat':sdgsht[y_cat[0]]})

if __name__ == '__main__':
    app.run(debug=True)