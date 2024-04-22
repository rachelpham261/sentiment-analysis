import regex as re
from langdetect import detect

import pandas as pd
import numpy as np
import torch
import random
from transformers import DistilBertModel, DistilBertTokenizer, BertModel, BertTokenizer
import joblib
import streamlit as st

# set seed
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

def clean_text(text):
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
    cleaned_text = cleaned_text.lower()
    return cleaned_text

def get_embeddings(sample_comments, model, tokenizer):
    sample_comments = [clean_text(comment) for comment in sample_comments]
    tokenized = [tokenizer.encode(comment, add_special_tokens=True) for comment in sample_comments]
    
    # pad to max length
    max_len = 0 # the maximum sequence length of the reviews
    for i, review in enumerate(tokenized):
        if len(review) > max_len:
            max_len = len(review)

    # pad the sequences to the maximum length
    padded = np.array([review + [0]*(max_len-len(review)) for i, review in enumerate(tokenized)])
    
    # get attn mask
    attention_mask = np.where(padded != 0, 1, 0) # 0 means ignore
    attention_mask = torch.tensor(attention_mask)
    input_ids = torch.tensor(padded)
    
    with torch.no_grad():
        last_hidden_states = model(input_ids, attention_mask=attention_mask)
    embeddings = last_hidden_states[0][:,0,:].numpy()
    return embeddings

def pipeline(comments, model_path):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    embeddings = get_embeddings(comments, model, tokenizer)
    model = joblib.load(model_path)
    predictions = model.predict(embeddings)
    sentiment_map = {-1: 'Negative', 1: 'Positive', 0: 'Neutral'}
    predictions = [sentiment_map[pred] for pred in predictions]
    prediction_df = pd.DataFrame({'comment': comments, 'sentiment': predictions})
    return prediction_df

def main():
    st.set_page_config("Movie Review Analysis")
    st.header("Know Your Audience From 🍅 to Reddit.")
    st.write("With the help of BERT LLM, you can predict the sentiment of movie reviews+. By collecting comments from Rotten Tomatoes Audiences, we can see how people react to the movies and predict the sentiment on Reddit comments.")
    st.write("In the future, we will upgrade this website to make you throw in a Reddit link and use it to predict the score for this movie from audiences!")

    user_input = st.text_input("Test the comment:","You could HEAR the disappointment from the audience I was with when that â€œTo Be Continuedâ€ title card showed up lmao. But besides that, yeah this was a dang fine sequel. Better than most I dare say. Seeing aged-up Peni Parker in her upgraded mech for the first time almost made me tear up a bit.Oh yeah and Hailee Steinfeld deserves an Oscar for her performance here, all of the scenes when she's paired up with her dad were beautifully executed. Easily the best parts of the film for me personally.")
    user_input = [user_input]

    if st.button("Predict"):
        result = pipeline(user_input, model_path='sentiment_analysis_model.pkl')
        st.success(f"The sentiment of the comment is: {result['sentiment'][0]}")

if __name__ == "__main__":
    main()