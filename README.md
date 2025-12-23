\# AI Support Ticket Classifier



An AI-powered support ticket classifier built in Python that predicts both \*\*ticket type\*\* and \*\*priority\*\* using Natural Language Processing (NLP).



The model is trained on \*\*28,000+ real customer support tickets\*\* and can be run directly from the terminal.



---



\## 🚀 Features

\- Predicts \*\*ticket category\*\* (e.g., Incident, Problem, Change)

\- Predicts \*\*ticket priority\*\* (low / medium / high)

\- Uses real-world enterprise-style data

\- Runs locally from the command line

\- Clean, modular Python code



---



\## 🧠 How It Works

1\. Ticket text is converted into numerical features using \*\*TF-IDF\*\*

2\. Two machine learning models are used:

&nbsp;  - Model 1: Ticket \*\*Type\*\* classifier

&nbsp;  - Model 2: Ticket \*\*Priority\*\* classifier

3\. Models are trained using \*\*Logistic Regression\*\*

4\. Trained models are saved and reused using `joblib`



---



\## 📊 Dataset

\- Source: Hugging Face public dataset

\- Size: \*\*28,260 English-language tickets\*\*

\- Fields used:

&nbsp; - `body` → ticket text

&nbsp; - `type` → ticket category

&nbsp; - `priority` → urgency level



---



\## 🛠 Tech Stack

\- Python 3.11

\- pandas

\- scikit-learn

\- joblib

\- Hugging Face `datasets`



---



\## ▶️ How to Run



\### 1. Clone the repository

```bash

git clone <repo-url>

cd ticket-classifier



