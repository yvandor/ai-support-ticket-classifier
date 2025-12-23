# AI Support Ticket Classifier

An AI-powered support ticket classifier built in Python that predicts both **ticket type** and **priority** using Natural Language Processing (NLP).

The model is trained on **28,260+ real English customer support tickets** and can be run directly from the terminal.

---

## 🚀 Features
- Predicts **ticket category/type** (e.g., Incident, Problem, Change)
- Predicts **ticket priority** (low / medium / high)
- Uses real-world enterprise-style data
- Runs locally from the command line
- Clean, modular Python code

---

## 🧠 How It Works
1. Ticket text is converted into numerical features using **TF-IDF**
2. Two machine learning models are used:
   - Model 1: Ticket **Type** classifier
   - Model 2: Ticket **Priority** classifier
3. Models are trained using **Logistic Regression**
4. Trained models are saved and reused using `joblib`

---

## 📊 Dataset
- Source: Hugging Face public dataset
- Size used: **28,260 English-language tickets**
- Fields used:
  - `body` → ticket text
  - `type` → ticket category
  - `priority` → urgency level

---

## 🛠 Tech Stack
- Python 3.11
- pandas
- scikit-learn
- joblib
- Hugging Face `datasets`

---

## ▶️ How to Run

### 1) Clone the repository
```bash
git clone https://github.com/yvandor/ai-support-ticket-classifier.git
cd ai-support-ticket-classifier
