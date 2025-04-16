# 🧠 Lifestyle Disease Risk Prediction Web App

This project is a **Flask-based AI web application** that predicts a user's lifestyle disease risk based on their body metrics and health data. It also provides personalized **diet and exercise recommendations** using **Google Gemini API**.

---

## 🚀 Features

- 🔍 **Risk Level Prediction** using a trained **Random Forest** model
- 🧮 **BMI, BFP, and Metabolic Age Calculators**
- 🔁 **Dynamic Caloric Intake Estimation**
- 🥗 **AI-generated Diet and Exercise Plans** with Gemini API
- 📊 Data pre-processing and model training using `pandas`, `scikit-learn`, `xgboost`
- 💾 Uses `pickle` to load trained models and features
- 🌐 Built with **Flask**, uses **HTML templates** for frontend

---

## 📂 Project Structure

```
├── app.py                  # Main Flask backend
├── rf_model.pkl            # Trained Random Forest model (must be added)
├── f_name.pkl              # Feature name list (must be added)
├── templates/
│   ├── index.html          # Main UI for prediction
│   ├── bfp.html            # BFP calculator page
│   └── diet.html           # Diet and exercise recommendation view
├── lifestyle_diseases_.ipynb # Jupyter Notebook used for model training
```

---

## 📈 How It Works

1. **User Input**: Age, gender, weight, height, activity level, BFP, heart rate.
2. **Feature Engineering**: Calculates BMI, metabolic age, and caloric intake.
3. **Prediction**: Uses a trained machine learning model to classify disease risk.
4. **Recommendation**: Uses Google Gemini API to generate personalized diet and fitness suggestions.

---

## 🧪 Tech Stack

- **Frontend**: HTML + Bootstrap (via templates)
- **Backend**: Flask, Python
- **ML Model**: Random Forest Classifier (`sklearn`)
- **API**: Google Gemini for health recommendations
- **Libraries**: `numpy`, `pandas`, `math`, `pickle`, `re`, `google.generativeai`

---

## 📦 Setup Instructions

1. **Clone the repo**
   ```bash
   git clone https://github.com/your-username/lifestyle-disease-risk-predictor.git
   cd lifestyle-disease-risk-predictor
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add the missing files**
   - `rf_model.pkl` - your trained model
   - `f_name.pkl` - your feature name list

4. **Run the app**
   ```bash
   python app.py
   ```

5. **Visit in browser**
   ```
   http://127.0.0.1:5000
   ```

---

## 🔐 API Configuration

To use the Gemini API, configure your API key:
```python
genai.configure(api_key="YOUR_GEMINI_API_KEY")
```

---

## 📊 Model Training (Jupyter Notebook)

Check out the `lifestyle_diseases_.ipynb` notebook for data preprocessing, feature engineering, visualization, model training and evaluation.

---

## 🧠 Sample Prediction Features

- Age
- Gender
- Activity Level
- Body Fat Percentage (BFP)
- Caloric Intake
- Heart Rate
- Metabolic Age
- BMI

---

## 📃 License

MIT License. Feel free to use and modify.

---

## 🙌 Acknowledgments

- [Google Gemini API](https://ai.google.dev/)
- [Flask](https://flask.palletsprojects.com/)
- [scikit-learn](https://scikit-learn.org/)
- [XGBoost](https://xgboost.readthedocs.io/)

---

## 👤 Author

**Arjun**  
AI/ML Developer | Python Enthusiast | Passionate about HealthTech  
📫 [LinkedIn or Email link here]
