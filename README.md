# 🛡️ Threat-Detecting AI Moderation Engine

🟢 **Live Web App:** [Play with the live BERT model here!](https://huggingface.co/spaces/CPU2516/threat-detecting-ai)

*(Note: Replace "YOUR_SPACE_NAME" with your actual Hugging Face URL before saving!)*

## 🧠 Model Architecture & Performance
* **Base Model:** BERT (Bidirectional Encoder Representations from Transformers)
* **Frameworks:** PyTorch, Hugging Face Transformers
* **Accuracy:** 89.5%
* **F1-Score:** 89.2%

## ⚠️ Known Limitations & Edge Cases
During rigorous red-team testing, the following edge cases were identified in the model's current iteration:
1. **Implicit Toxicity (False Negatives):** The model excels at detecting explicit profanity and threats, but occasionally struggles with highly professional, passive-aggressive insults that lack standard toxic vocabulary.
2. **Reclaimed Slang (False Positives):** The model occasionally flags aggressive compliments (e.g., "You completely killed that presentation!") as toxic due to the heavy negative weighting of violent verbs in the Jigsaw training dataset. 

## 🐳 Running Locally with Docker
1. Clone this repository.
2. Download the pre-trained model weights from the Hugging Face Space and place them inside the `bert-toxic-model/` directory.
3. Build the image: `docker build -t threat-analyzer .`
4. Run the container: `docker run -p 8501:8501 threat-analyzer`
