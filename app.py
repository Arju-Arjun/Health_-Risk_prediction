from flask import Flask, render_template, request, redirect, url_for, session
import pickle
import numpy as np
import pandas as pd
import math
import google.generativeai as genai

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load the trained model and feature names
model = pickle.load(open('rf_model.pkl', 'rb'))
f_name = pickle.load(open('f_name.pkl', 'rb'))

# Set up Google Gemini API client (replace with your actual API key)
genai.configure(api_key="AIzaSyBwm0QFEnBwHP7Qv2Hds8b-mcdQwOJTBFg")

def calculate_bfp(gender, waist, neck, height, hip=None):
    try:
        if gender == "male":
            if waist - neck <= 0 or height <= 0:
                return None
            return 86.01 * math.log10(waist - neck) - 70.041 * math.log10(height) + 36.76
        elif gender == "female":
            if waist + hip - neck <= 0 or height <= 0:
                return None
            return 163.205 * math.log10(waist + hip - neck) - 97.684 * math.log10(height) - 78.387
    except ValueError:
        return None

import re
import re

import re

import re

import re

def get_diet_exercise_from_api(age, risk):
    try:
        prompt = f"""
        Provide a diet and exercise recommendation for a person who is {age} years old with a {risk} health risk level.
        - For diet, suggest a concise meal plan or guidelines, including sample meals for **Breakfast**, **Lunch**, **Dinner**, and **Snacks** marked with * (e.g., * **Breakfast**: ...). Include multiple options in descriptions separated by commas or "or".
        - For exercise, list 2-3 specific activities with brief descriptions (no links needed), with only the activity names bolded (e.g., **Activity 1**). Include multiple options in descriptions separated by commas or "or". Ensure proper spacing and hyphenation (e.g., "Low-impact" not "Lowimpact").
        Return the response in this format:
        Diet: [Your diet recommendation with sample meals]
        Exercise: 
        - [**Activity 1**]: [Description]
        - [**Activity 2**]: [Description]
        - [**Activity 3**]: [Description]
        Use ** to bold only the subheadings (**Breakfast**, **Lunch**, **Dinner**, **Snacks** in diet, and activity names in exercise). Do not bold any other words or phrases within the text.
        """

        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)

        raw_response = response.text.strip()
        print("Raw API Response:", raw_response)

        # Convert **text** to <strong>text</strong>
        processed_text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', raw_response)
        print("Processed Text:", processed_text)

        # Split into lines
        lines = processed_text.split('\n')
        
        diet = ""
        exercises = []
        current_exercise = ""
        in_exercise_section = False

        for line in lines:
            line = line.strip()
            if line.startswith("Diet:"):
                diet = line.replace("Diet:", "").strip()
            elif line.startswith("Exercise:"):
                in_exercise_section = True
            elif in_exercise_section and line.startswith("-"):
                if current_exercise:
                    exercises.append(current_exercise)
                current_exercise = line.replace("-", "").strip()
            elif in_exercise_section and line:
                current_exercise += " " + line.strip()
            elif line.startswith("*"):  # Handle meal items under diet
                diet += "<br>" + line.replace("*", "").strip()

        if current_exercise:
            exercises.append(current_exercise)

        print("Diet:", diet)
        print("Exercises:", exercises)

        return diet, exercises

    except Exception as e:
        return f"Error fetching recommendation: {str(e)}", []
    
    
    
def calorie_calculator(weight, height, age, gender, activity):
    if gender.lower() == "male":
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161

    activity_levels = {
        "Sedentary": 1.2,
        "Lightly Active": 1.375,
        "Moderately Active": 1.55,
        "Active": 1.725
    }

    return round(bmr * activity_levels.get(activity, 1.2), 2)

def metabolic_age(weight, height, age, gender):
    gender = gender.lower()
    
    avg_bmr_table = {
        "male": {20: 1850, 30: 1750, 40: 1700, 50: 1650, 60: 1600, 70: 1550},
        "female": {20: 1550, 30: 1450, 40: 1400, 50: 1350, 60: 1300, 70: 1250}
    }

    if gender not in avg_bmr_table:
        raise ValueError(f"Invalid gender: {gender}")

    if not (18 <= age <= 70):
        raise ValueError(f"Age {age} is out of range for metabolic age calculation")

    bmr = (10 * weight + 6.25 * height - 5 * age + (5 if gender == "male" else -161))
    
    avg_bmr = avg_bmr_table[gender][min(avg_bmr_table[gender], key=lambda x: abs(x - age))]
    
    return round((bmr / avg_bmr) * age, 2)

@app.route('/')
def home():
    bfp_value = request.args.get('bfp', '') 
    height = request.args.get('height', session.get('height', '')) 
    gender = request.args.get('gender', session.get('gender', 'male')).title() 
    return render_template('index.html', prediction_text='', bfp=bfp_value, height=height, gender=gender)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        Age = int(request.form['age'])
        Height = float(request.form['height']) / 100  # Convert to meters
        Weight = float(request.form['weight'])
        Gender = request.form['gender']
        Activity_Level = request.form['activity']
        Body_Fat_Percentage = float(request.form['fat'])
        Heart_Rate = float(request.form['heart_rate'])
        Metabolic_Age = float(metabolic_age(Weight, Height, Age, Gender))

        # Calculate additional features
        BMI = round(Weight / (Height ** 2), 2)
        Caloric_Intake = calorie_calculator(Weight, Height * 100, Age, Gender, Activity_Level)

        # Prepare input data
        input_data = np.array([[Age, Gender, Activity_Level, Body_Fat_Percentage, Caloric_Intake, Heart_Rate, Metabolic_Age, BMI]])
        new_df = pd.DataFrame(input_data, columns=f_name)

        # Encode categorical values
        activity_mapping = {'Sedentary': 0, 'Lightly Active': 1, 'Moderately Active': 2, 'Active': 3}
        gender_mapping = {'Female': 0, 'Male': 1}

        if "Activity Level" in new_df.columns:
            new_df["Activity Level"] = new_df["Activity Level"].map(activity_mapping)
        if "Gender" in new_df.columns:
            new_df["Gender"] = new_df["Gender"].map(gender_mapping)

        prediction = model.predict(new_df)

        risk_levels = {0: 'Low', 1: 'Medium', 2: 'High', 3: 'Very High', 4: 'Severe'}
        prediction_text = risk_levels.get(prediction[0], 'Error')

        session['age'] = Age
        session['risk'] = prediction_text

        return render_template('index.html', gender="", prediction_text=prediction_text)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

@app.route('/calculate', methods=['GET', 'POST'])
def calculate():
    if request.method == 'POST':
        gender = request.form.get('gender')
        waist = request.form.get('waist')
        neck = request.form.get('neck')
        height = request.form.get('height')
        hip = request.form.get('hip') if gender == "female" else None

        try:
            waist_f = float(waist)
            neck_f = float(neck)
            height_f = float(height)
            hip_f = float(hip) if hip else None

            bfp = calculate_bfp(gender, waist_f, neck_f, height_f, hip_f)

            if isinstance(bfp, (int, float)):
                session['height'] = height 
                session['gender'] = gender.title()
                return redirect(url_for('home', bfp=f"{bfp:.2f}", height=height, gender=gender))  
            else:
                return render_template('bfp.html', bfp=bfp, gender=gender, waist=waist, neck=neck, height=height, hip=hip)

        except ValueError:
            return render_template('bfp.html', bfp="Invalid Input", gender=gender, waist=waist, neck=neck, height=height, hip=hip)

    return render_template('bfp.html', bfp=None)

@app.route('/diet_exercise')
def diet():
    age = session.get('age')
    risk = session.get('risk')

    if not age or not risk:
        return render_template('diet.html', diet="No recommendation available", exercise="No recommendation available")

    # Fetch diet and exercise recommendations from Google Gemini API
    diet, exercise = get_diet_exercise_from_api(age, risk)

    return render_template('diet.html', diet=diet, exercise=exercise)

if __name__ == '__main__':
    app.run(debug=True)