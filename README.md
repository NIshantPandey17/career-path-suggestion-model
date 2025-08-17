# 🎓 Career Path Suggestion System  

An AI-powered web application that helps B.Tech students explore the most suitable career paths based on their academic performance, technical skills, and interests.  

---

## 🚀 Features  
- Generates a synthetic dataset for career path prediction.  
- Trains a **Random Forest Classifier** model.  
- Interactive **Streamlit web app** to predict careers.  
- Considers multiple factors like:  
  - CGPA  
  - Programming skills  
  - Problem-solving skills  
  - Communication, leadership, and creativity  
  - Specialized knowledge (AI/ML, Networking, Cloud, Blockchain, etc.)  
- Suggests career roles such as:  
  **Software Engineer, Data Scientist, AI/ML Engineer, Cybersecurity Specialist, Cloud Engineer, UI/UX Designer, Project Manager** and more.  

---

## 🛠 Tech Stack  
- **Python 3.9+**  
- **Pandas, NumPy** → Data handling  
- **Scikit-learn** → ML model training  
- **Streamlit** → Web application  
- **Pickle** → Model persistence  

---

## 📂 Project Structure  
```
Career-Path-Suggestion/
│── career_suggestion.py # Main Streamlit app
│── BTech_Career_Path_Dataset.csv # Generated dataset
│── career_suggestion.pkl # Trained ML model
│── README.md # Project documentation
│── requirements.txt # Dependencies

```
## ⚙️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/face-recognition-app.git
cd face-recognition-app
```
2️⃣ Install dependencies
```bash

pip install -r requirements.txt
```
3️⃣ Run the app locally
```bash

streamlit run app.py
```
