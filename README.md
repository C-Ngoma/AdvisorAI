# 🎓 Career Path Advisory AI

> *“Your AI-powered career counselor — matching students to real-world opportunities.”*



## 📌 Project Overview

Career Path Advisory AI is a system that recommends career paths to students based on their academic background, skills, and job market trends.
It uses **NLP, time series forecasting, and deep learning** to provide interactive career guidance.
Developed by a group of 10 dedicated students who witnessed a significant rise in unemployment, AdvisorAI aims to empower individuals. Our mission is to bridge the gap between opportunity and talent, offering actionable advice and guidance to help users achieve their goals. 


---

#🚀 Features

* 🗣️ **NLP Career Queries** → Answer questions like *“What can I do with a degree in Economics?”*
* 📈 **Salary & Demand Forecasting** → Predict job demand and salary trends using time series analysis
* 🤖 **Resume & Skill Matching** → Match a student’s skills with job requirements and recommend roles
* 💬 **Chatbot Interface** → Interactive guidance for career planning

---

# 🛠️ Tech Stack

* **Languages:** Python 
* **Libraries & Tools:**

  * Data: Pandas, NumPy
  * NLP: Scikit-learn (TF-IDF, Cosine Similarity)
  * Forecasting: Prophet / ARIMA
  * Deep Learning: TensorFlow / PyTorch
  * Frontend: React / HTML / CSS
  * Backend: Flask / FastAPI
* **Version Control:** Git & GitHub

---

# 📂 Project Structure

```
AdvisorAI/
│── datacollection/         # Datasets (raw, cleaned, merged)
│── src/                    # Source code
│   │── skill_matcher.py    # Resume & Skill matching engine
│   │── data_cleaning.py    # Dataset cleaning script
│── docs/                   # Documentation & report files
│── LICENSE                 # Project license (MIT)
│── README.md               # Project documentation
```

---

#🧪 How It Works

1. **Input:** Student enters skills or degree
2. **Processing:**

   * NLP → Understands the query
   * Skill Matcher → Compares input with job dataset
   * Forecasting → Predicts future salaries/demand
3. **Output:** Suggested careers, salary range, required skills

---

# 📊 Sample Output (Resume Matcher)

**Input Skills:** `python, sql, machine learning`

**Recommended Roles:**

| Job Title      | Years Exp | Salary (USD) | Match % |
| -------------- | --------- | ------------ | ------- |
| Data Scientist | 3         | 152,626      | 87.5%   |
| AI Specialist  | 2         | 55,770       | 78.3%   |
| ML Engineer    | 1         | 69,893       | 75.4%   |

---

# 👥 Team Roles 

* 📊 **Data Preparation & Cleaning** → Christina 
* 🗣️ **NLP Queries** → Mmabatho 
* 📈 **Time Series Forecasting** → Sabelo 
* 🤖 **Deep Learning & Resume Parsing** → 
* ⚙️ **Backend & API Integration** → Pitsi
  



---

# 📜 License

This project is licensed under the [MIT License](inprogess).

---

# 🙌 Acknowledgements

* Kaggle – Job Skills & Salary Data
* Scikit-learn & Pandas Community
* VUT (Project 2025)



