import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

#1. Build the ui
st.set_page_config(page_title="Toxicity Classifier", page_icon="🛡️")
st.title("🛡️ Threat-Detection AI Model")
st.markdown("Enter a comment below to evaluate it for threats, insults or hate speech")

# 2. Load the model and cache it so it doesn't reload on every interaction
@st.cache_resource
def load_model():
    # Dynamically find the path: gets the exact location of this file (/src)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Goes up one level to the root, then into the model folder
    model_path = os.path.join(current_dir, "..", "bert-toxic-model")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model

tokenizer, model = load_model()



user_input = st.text_area("User Comment", placeholder="Type a sentence here...")

# 3. Handle the prediction logic here
if st.button("Analyze Text"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze")
    else:
        with st.spinner('Analyzing...'):
            # Tokenize and predict
            inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)

            # Calculate probabilities
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred_idx = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][pred_idx].item() * 100

            # Display Result
            st.divider()
            if pred_idx == 1:
                st.error(f"**Prediction: Toxic** ⚠️ (Confidence: {confidence:.2f}%)")
            else:
                st.success(f"**Prediction: Non-Toxic** ✅ (Confidence: {confidence:.2f}%)")
