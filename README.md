# 🛡️ Advai Prompt Guard: Detecting Malicious LLM Prompts

This Streamlit app is a hands-on demo of adversarial prompt detection and model retraining, designed to simulate how malicious prompts can bypass AI safety filters — and how models can be hardened in response.

---

## 🔍 What This App Does

- Detects malicious prompts using a logistic regression classifier
- Generates **adversarial examples** from safe prompts to simulate attacks
- Allows **user feedback and retraining** for real-time model improvement
- Visualizes key features influencing the classifier

---

## 🚀 Try It Live

You can enter your own prompt and see how the model classifies it as **Safe** or **Malicious** — and if it's wrong, you can correct it and **instantly retrain** the model.

---

## 🧠 Features

| Feature                        | Description                                                              |
|-------------------------------|--------------------------------------------------------------------------|
| ✅ Prompt Classification       | Labels text as safe or malicious using a logistic regression model       |
| 🔁 User Feedback Loop         | Allows retraining the model with user-labeled examples in real-time      |
| 🧬 Adversarial Prompt Gen     | Mutates safe prompts into malicious variants to simulate bypasses        |
| 📊 Interpretability Tools     | View top weighted words influencing malicious classification              |

---

## 🛠️ Technologies Used

- Streamlit
- scikit-learn
- TF-IDF vectorization
- Logistic Regression
- Matplotlib

---

## 📦 Installation

```bash
git clone https://github.com/your-username/prompt-guard-demo.git
cd prompt-guard-demo
pip install -r requirements.txt
streamlit run app.py
