import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

#1. Load the model and cache it so it dosent reload on every button click
def load_model():
    #Pointing to the folder
    model_path = "./bert-toxic-model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model

tokenizer,model = load_model()

#2. Build the ui
st.set_page_config(page_title="Toxicity Classifier", page_icon="https://www.svgrepo.com/show/527879/shield-minimalistic.svg")
st.title("Threat-Detection AI Model")
st.markdown("Enter a comment below to evaluate it for threats, insults or hate speech")

user_input = st.text_area("User Comment", placeholder="Type a sentence here...")

#3. Handle the prediction logic here
if st.button("Analyze Text"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze")
    else:
        with st.spinner('Analyzing...'):
            #Tokenize and predict
            inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)

                #Calculate probabilities
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred_idx = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][pred_idx].item()*100

            #Display Result
            st.divider()
            if pred_idx == 1:
                st.error(f"**Prediction: Toxic** (Confidence: {confidence:.2f}%)")
            else:
                st.success(f"**Prediction: Non-Toxic** (Confidence: {confidence:.2f}%)")
