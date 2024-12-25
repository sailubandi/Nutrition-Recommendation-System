import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from joblib import dump, load
import os

def clean_numeric(value):
    if isinstance(value, str):
        return float(''.join(filter(lambda x: x.isdigit() or x == '.', value)))
    return value

@st.cache_data
def load_and_prepare_data():
    data = pd.read_excel('nutrition.xlsx')
    numeric_columns = ['calories', 'protein', 'carbohydrate', 'total_fat', 'fiber']
    
    for col in numeric_columns:
        if col not in data.columns:
            data[col] = 0
        data[col] = data[col].apply(clean_numeric)

    features = data[['calories', 'protein', 'carbohydrate', 'total_fat']]
    kmeans = KMeans(n_clusters=3, random_state=42)
    data['meal_type'] = kmeans.fit_predict(features)
    cluster_map = dict(enumerate(['breakfast', 'lunch', 'dinner']))
    data['meal_type'] = data['meal_type'].map(cluster_map)

    return data

def train_random_forest(data):
    features = ['calories', 'protein', 'carbohydrate', 'total_fat', 'fiber']
    X = data[features]
    y = (data['calories'] > data['calories'].mean()).astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    y_pred = rf_model.predict(X)
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred),
        'recall': recall_score(y, y_pred),
        'f1': f1_score(y, y_pred),
    }
    dump(rf_model, 'rf_model.joblib')
    
    return rf_model, y_pred, metrics

def get_bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 25:
        return "Normal"
    elif 25 <= bmi < 30:
        return "Overweight"
    else:
        return "Obese"

def get_recommendations(data, rf_model, bmi, goal, meal_preference, y_pred, num_recommendations=5):
    filtered_data = data[data['meal_type'] == meal_preference.lower()].copy()
    filtered_data['is_high_calorie'] = (y_pred[filtered_data.index] == 1)
    bmi_category = get_bmi_category(bmi)

    if bmi_category == "Underweight":
        if goal == "Weight Loss":
            filtered_data = filtered_data[(filtered_data['calories'] >= filtered_data['calories'].mean() * 0.75) &
                                          (filtered_data['calories'] <= filtered_data['calories'].mean() * 1.25)]
        elif goal == "Weight Gain":
            filtered_data = filtered_data[filtered_data['calories'] > filtered_data['calories'].mean()]
        elif goal == "Healthy":
            filtered_data = filtered_data[filtered_data['calories'] > filtered_data['calories'].mean()]
    elif bmi_category == "Normal":
        if goal == "Weight Loss":
            filtered_data = filtered_data[filtered_data['calories'] < filtered_data['calories'].mean()]
        elif goal == "Weight Gain":
            filtered_data = filtered_data[(filtered_data['calories'] >= filtered_data['calories'].mean() * 0.75) &
                                          (filtered_data['calories'] <= filtered_data['calories'].mean() * 1.25)]
        elif goal == "Healthy":
            filtered_data = filtered_data[filtered_data['calories'] > filtered_data['calories'].mean()]
    elif bmi_category == "Overweight":
        if goal == "Weight Loss":
            filtered_data = filtered_data[(filtered_data['calories'] >= filtered_data['calories'].mean() * 0.5) & 
                                          (filtered_data['calories'] < filtered_data['calories'].mean())]
        elif goal == "Weight Gain":
                        filtered_data = filtered_data[(filtered_data['calories'] >= filtered_data['calories'].mean() * 0.75) & 
                                          (filtered_data['calories'] <= filtered_data['calories'].mean() * 1.25)]
        elif goal == "Healthy":
            filtered_data = filtered_data[filtered_data['calories'] < filtered_data['calories'].mean()]
    elif bmi_category == "Obese":
        if goal == "Weight Loss":
            filtered_data = filtered_data[filtered_data['calories'] < filtered_data['calories'].mean()]
        elif goal == "Weight Gain":
            filtered_data = filtered_data[(filtered_data['calories'] >= filtered_data['calories'].mean() * 0.75) & 
                                          (filtered_data['calories'] <= filtered_data['calories'].mean() * 1.25)]
        elif goal == "Healthy":
            filtered_data = filtered_data[filtered_data['calories'] < filtered_data['calories'].mean()]

    sorted_data = filtered_data.sort_values('calories', ascending=False)
    recommendations = sorted_data.head(num_recommendations)

    recommendation_details = []
    for _, food_info in recommendations.iterrows():
        recommendation_details.append({
            'name': food_info['name'],
            'calories': food_info['calories'],
            'protein': food_info['protein'],
            'carbs': food_info['carbohydrate'],
            'fat': food_info['total_fat'],
            'fiber': food_info['fiber']
        })

    return recommendation_details

def user_input_page(goal):
    st.title(f"🥗 {goal} Recommendation")

    st.subheader("📝 Personal Information")
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age (years)", min_value=0, value=25, step=1)
        weight = st.number_input("Weight (kg)", min_value=0, value=70, step=1)
        height = st.number_input("Height (m)", min_value=0.1, value=1.75, step=0.01)

    with col2:
        meal_preference = st.selectbox("Select Meal Type", ["Breakfast", "Lunch", "Dinner"])

    bmi = weight / (height ** 2)
    bmi_category = get_bmi_category(bmi)

    data = load_and_prepare_data()

    st.subheader("🏥 Health Status")
    st.write(f"Your BMI: {bmi:.2f}")
    st.write(f"BMI Category: {bmi_category}")

    if bmi_category == "Underweight":
        st.warning("You are Underweight")
    elif bmi_category == "Normal":
        st.success("You are in the Healthy range")
    elif bmi_category == "Overweight":
        st.warning("You are Overweight")
    else:
        st.error("You are in the Obese range")

    if st.button("Get Recommendations"):
        display_recommendations(bmi, bmi_category, data, meal_preference, goal)

    if st.button("🏠 Return to Home"):
        st.session_state.page = "home"
        st.rerun()

def display_recommendations(bmi, bmi_category, data, meal_preference, goal):
    rf_model, y_pred, metrics = train_random_forest(data)
    recommendations = get_recommendations(data, rf_model, bmi, goal, meal_preference, y_pred)

    st.subheader(f"🍽️ Recommended {meal_preference} Foods")
    cols = st.columns(3)
    for idx, food in enumerate(recommendations):
        col = cols[idx % 3]
        with col:
            st.markdown(f"""
            **{idx + 1}. {food['name']}**
            - Calories: {food['calories']:.0f} kcal
            - Protein: {food['protein']:.1f}g
            - Carbs: {food['carbs']:.1f}g
            - Fat: {food['fat']:.1f}g
            - Fiber: {food['fiber']:.1f}g
            """)

    st.subheader("💡 Meal-Specific Tips")
    meal_tips = {
        "Breakfast": """
        • Start your day with protein-rich foods
        • Include complex carbohydrates for sustained energy
        • Add fruits for essential vitamins and fiber
        • Don't skip breakfast - it's crucial for metabolism
        """,
        "Lunch": """
        • Balance your plate with protein, carbs, and vegetables
        • Keep portions moderate to maintain energy levels
        • Include a variety of colorful vegetables
        • Stay hydrated throughout the meal
        """,
                "Dinner": """
        • Choose lighter options if eating late
        • Include lean proteins and vegetables
        • Limit heavy carbohydrates in the evening
        • Allow 2-3 hours between dinner and bedtime
        """
    }

    st.write(meal_tips[meal_preference])

    # Goal-specific tips
    if goal == "Weight Loss":
        tips = """
        • Control portion sizes
        • Choose foods high in protein and fiber
        • Limit added sugars and processed foods
        • Stay hydrated throughout the day
        """
    elif goal == "Weight Gain":
        tips = """
        • Increase portion sizes gradually
        • Focus on nutrient-dense foods
        • Add healthy fats to your meals
        • Eat more frequently throughout the day
        """
    else:  # Healthy
        tips = """
        • Maintain balanced portions
        • Include a variety of food groups
        • Listen to your hunger cues
        • Stay consistent with meal timing
        """

    st.write(tips)

def home_page():
    st.title("🥗 Nutrition Recommendation System")
    st.write("Welcome to the Nutrition Recommendation System!")
    st.write("Please select your goal:")

    st.subheader("Weight Loss")
    st.markdown("""
        <p style='text-align: justify;'>
        Achieve healthy weight loss with balanced, lower-calorie meal options tailored 
        to help you shed pounds sustainably.
        </p>
    """, unsafe_allow_html=True)
    if st.button("Weight Loss"):
        st.session_state.page = "weight_loss"
        st.rerun()
        
    st.subheader("Weight Gain")
    st.markdown("""
        <p style='text-align: justify;'>
        Reach your weight gain goals with nutrient-rich, calorie-dense meal plans 
        designed to help you build muscle and strength.
        </p>
    """, unsafe_allow_html=True)
    if st.button("Weight Gain"):
        st.session_state.page = "weight_gain"
        st.rerun()

    st.subheader("Healthy Living")
    st.markdown("""
        <p style='text-align: justify;'>
        Enjoy a balanced diet rich in essential nutrients for overall health, boosting energy 
        and improving well-being.
        </p>
    """, unsafe_allow_html=True)
    if st.button("Healthy Living"):
        st.session_state.page = "healthy"
        st.rerun()

    st.subheader("About Our Programs:")
    st.write("**Weight Loss Program** 🏋️‍♀️")
    st.write("Designed for those looking to shed pounds in a healthy and sustainable way.")
    st.write("**Weight Gain Program** 💪")
    st.write("Perfect for individuals aiming to build muscle and increase body mass.")
    st.write("**Healthy Living Program** 🥗")
    st.write("Balanced nutrition advice for maintaining optimal health and wellness.")


def main():
    st.write("Current working directory:", os.getcwd())
    model_path = 'rf_model.joblib'
    if os.path.exists(model_path):
        st.write("Model file found.")
    else:
        st.write("Model file not found. It will be trained.")
    
    if 'page' not in st.session_state:
        st.session_state.page = "home"

    if 'data' not in st.session_state:
        st.session_state.data = load_and_prepare_data()

    if 'rf_model' not in st.session_state:
        try:
            st.session_state.rf_model = load('rf_model.joblib')
        except FileNotFoundError:
            st.session_state.rf_model, _, st.session_state.model_metrics = train_random_forest(st.session_state.data)

    if st.session_state.page == "home":
        home_page()
    elif st.session_state.page == "weight_loss":
        user_input_page("Weight Loss")
    elif st.session_state.page == "weight_gain":
        user_input_page("Weight Gain")
    elif st.session_state.page == "healthy":
        user_input_page("Healthy Living")
        
if __name__ == "__main__":
    main()