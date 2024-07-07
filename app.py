import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('diabetes.csv')

# HEADINGS
st.title('Diabetes Checkup')
st.sidebar.header('Patient Data')
st.subheader('Training Data Stats')
st.write(df.describe())

# X AND Y DATA
x = df.drop(['Outcome'], axis=1)
y = df['Outcome']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# FUNCTION
def user_report():
    pregnancies = st.sidebar.slider('Pregnancies', 0, 7, 3)
    glucose = st.sidebar.slider('Glucose', 0, 200, 120)
    bp = st.sidebar.slider('Blood Pressure', 0, 122, 70)
    skinthickness = st.sidebar.slider('Skin Thickness', 0, 100, 20)
    insulin = st.sidebar.slider('Insulin', 0, 846, 79)
    bmi = st.sidebar.slider('BMI', 0, 67, 20)
    dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.4, 0.47)
    age = st.sidebar.slider('Age', 21, 88, 33)

    user_report= {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': bp,
        'SkinThickness': skinthickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }
    report_data = pd.DataFrame(user_report, index=[0])
    return report_data

# PATIENT DATA
user_data = user_report()
st.subheader('Patient Data')
st.write(user_data)
# MODEL
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
user_result = rf.predict(user_data)

# VISUALISATIONS
st.title('Visualised Patient Report')

# COLOR FUNCTION
color = 'blue' if user_result[0] == 0 else 'red'

# Visualizations
def plot_graph(x, y, hue, palette, user_x, user_y, title):
    fig = plt.figure()
    ax = sns.scatterplot(x=x, y=y, data=df, hue=hue, palette=palette)
    sns.scatterplot(x=user_x, y=user_y, s=150, color=color)
    plt.title(title)
    st.pyplot(fig)

plot_graph('Age', 'Pregnancies', 'Outcome', 'Greens', user_data['Age'], user_data['Pregnancies'], 'Pregnancy count Graph (Others vs Yours)')
plot_graph('Age', 'Glucose', 'Outcome', 'magma', user_data['Age'], user_data['Glucose'], 'Glucose Value Graph (Others vs Yours)')
plot_graph('Age', 'BloodPressure', 'Outcome', 'Reds', user_data['Age'], user_data['BloodPressure'], 'Blood Pressure Value Graph (Others vs Yours)')
plot_graph('Age', 'SkinThickness', 'Outcome', 'Blues', user_data['Age'], user_data['SkinThickness'], 'Skin Thickness Value Graph (Others vs Yours)')
plot_graph('Age', 'Insulin', 'Outcome', 'rocket', user_data['Age'], user_data['Insulin'], 'Insulin Value Graph (Others vs Yours)')
plot_graph('Age', 'BMI', 'Outcome', 'rainbow', user_data['Age'], user_data['BMI'], 'BMI Value Graph (Others vs Yours)')
plot_graph('Age', 'DiabetesPedigreeFunction', 'Outcome', 'YlOrBr', user_data['Age'], user_data['DiabetesPedigreeFunction'], 'DPF Value Graph (Others vs Yours)')

# OUTPUT
st.subheader('Your Report: ')
output = 'You are Diabetic' if user_result[0] == 1 else 'You are not Diabetic'
st.title(output)

# ACCURACY
st.subheader('Accuracy: ')
st.write(f'{accuracy_score(y_test, rf.predict(x_test)) * 100:.2f}%')

# MEDICAL ADVICE
st.subheader('Medical Advice: ')

def provide_advice(glucose, bp, insulin, bmi, dpf, age):
    advice = []

    if glucose > 125:
        advice.append("Your glucose level is high. It's recommended to follow a low-sugar diet and consult with a healthcare provider.")
    elif glucose < 70:
        advice.append("Your glucose level is low. Ensure you have regular meals and snacks. Consult a healthcare provider for personalized advice.")
    
    if bp > 120:
        advice.append("Your blood pressure is high. Regular exercise and a low-sodium diet can help manage blood pressure. Consult your healthcare provider.")
    
    if insulin > 200:
        advice.append("Your insulin level is high. It's essential to manage carbohydrate intake and discuss with your healthcare provider for insulin management.")
    elif insulin < 20:
        advice.append("Your insulin level is low. Consult your healthcare provider for a detailed assessment and management plan.")
    
    if bmi > 25:
        advice.append("Your BMI is high, indicating overweight or obesity. Regular physical activity and a balanced diet are recommended. Consult a nutritionist or healthcare provider.")
    elif bmi < 18.5:
        advice.append("Your BMI is low. Consider a balanced diet rich in nutrients. Consult a nutritionist or healthcare provider for personalized advice.")
    
    if dpf > 1.0:
        advice.append("Your Diabetes Pedigree Function is high, indicating a higher genetic risk. Regular monitoring and a healthy lifestyle are crucial. Consult a healthcare provider.")
    
    if age > 45:
        advice.append("Age is a significant risk factor for diabetes. Regular screenings and a healthy lifestyle are recommended. Consult your healthcare provider for regular check-ups.")

    if not advice:
        advice.append("Your input values are within normal ranges. Maintain a healthy lifestyle with regular exercise, a balanced diet, and regular check-ups.")

    return advice

advice = provide_advice(user_data['Glucose'][0], user_data['BloodPressure'][0], user_data['Insulin'][0], user_data['BMI'][0], user_data['DiabetesPedigreeFunction'][0], user_data['Age'][0])
for line in advice:
    st.write(line)
