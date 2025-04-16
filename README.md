Health Risk Predictor

Description

Health Risk Predictor is a Flask-based web application designed to assess an individual's health risk level based on metrics like age, weight, height, body fat percentage, and activity level. It uses a Random Forest model trained on a BMI dataset to predict health risks (Low, Medium, High, Very High, Severe). The application also calculates Body Fat Percentage (BFP), daily caloric needs, and metabolic age, and provides personalized diet and exercise recommendations using the Google Gemini API.

Features





Health Risk Prediction: Predicts health risk levels using a pre-trained Random Forest model.



Body Fat Percentage Calculator: Calculates BFP based on gender-specific measurements (waist, neck, height, and hip for females).



Caloric Intake Estimation: Computes daily caloric needs based on weight, height, age, gender, and activity level.



Metabolic Age Calculation: Estimates metabolic age using BMR compared to age-specific averages.



Personalized Recommendations: Fetches diet and exercise plans from the Google Gemini API based on age and risk level.



User-Friendly Interface: Interactive web interface built with Flask and HTML templates.

Installation





Clone the repository:

git clone https://github.com/your-username/health-risk-predictor.git
cd health-risk-predictor



Create a virtual environment and activate it:

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate



Install dependencies:

pip install -r requirements.txt



Ensure the pre-trained model (rf_model.pkl) and feature names (f_name.pkl) are in the project directory.



Set up the Google Gemini API key:





Replace the API key in app.py (genai.configure(api_key="YOUR_API_KEY")) with your own key.

Usage





Run the Flask application:

python app.py



Open your browser and navigate to http://127.0.0.1:5000.



Use the web interface to:





Calculate Body Fat Percentage by entering measurements.



Predict health risk by inputting age, weight, height, gender, activity level, body fat percentage, and heart rate.



View personalized diet and exercise recommendations based on your inputs.

Project Structure

health-risk-predictor/
├── app.py                   # Main Flask application
├── rf_model.pkl             # Pre-trained Random Forest model
├── f_name.pkl               # Feature names for the model
├── templates/               # HTML templates (index.html, bfp.html, diet.html)
├── static/                  # Static files (CSS, JavaScript, images)
├── lifestyle_diseases_.ipynb # Jupyter Notebook for data analysis and model training
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation

Technologies Used





Languages: Python



Framework: Flask



Machine Learning: Scikit-learn (Random Forest), Pandas, NumPy



API: Google Gemini API for diet and exercise recommendations



Frontend: HTML, CSS, JavaScript (via templates)



Data Analysis: Jupyter Notebook, Matplotlib, Seaborn



Other Tools: Pickle (for model serialization)

Requirements





Python 3.8+



Flask>=2.0.0



scikit-learn>=1.0.0



pandas>=1.3.0



numpy>=1.19.0



google-generativeai>=0.3.0



See requirements.txt for the full list of dependencies.

Model Training

The Random Forest model was trained on a BMI dataset (BMI.csv) with features including Age, Gender, Activity Level, Body Fat Percentage, Daily Caloric Intake, Heart Rate, Metabolic Age, and BMI. The model achieved an accuracy of approximately 91.5% on the test set. The training process is detailed in lifestyle_diseases_.ipynb, which includes:





Data preprocessing (handling missing values, outlier removal using IQR).



Model evaluation with Logistic Regression, Decision Tree, Random Forest, XGBoost, and SVM.



Hyperparameter tuning for Random Forest (e.g., n_estimators=250, max_depth=23).

Contributing





Fork the repository.



Create a new branch (git checkout -b feature-branch).



Make your changes and commit (git commit -m 'Add some feature').



Push to the branch (git push origin feature-branch).



Open a Pull Request.

License

This project is licensed under the MIT License.

Acknowledgments





Dataset: BMI dataset used for training the model.



Google Gemini API for providing diet and exercise recommendations.



Scikit-learn for machine learning utilities.
