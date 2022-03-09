import numpy as np
import pickle
import streamlit as st

# Loading the saved model

loaded_model_lr = pickle.load(open('trained_model_lr.sav', 'rb'))
loaded_model_xgb = pickle.load(open('trained_model_xgb.sav', 'rb'))
loaded_model_cat = pickle.load(open('trained_model_cat.sav', 'rb'))

# creating a function for prediction

clf_name = st.sidebar.selectbox("Select Classifier Model: ",
                                ("Logistic Regression", "XGRBF Classifier", "Catboost Classifier"))


def pcos_prediction(input_data):

    input_data_as_numpy_array = np.asarray(input_data)

    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    if clf_name == 'Logistic Regression':
        prediction = loaded_model_lr.predict(input_data_reshaped)
        if prediction == 0:
            return "NO PCOS"
        else:
            return "PCOS"
    elif clf_name == 'XGRBF Classifier':
        prediction = loaded_model_xgb.predict(input_data_reshaped)
        if prediction == 0:
            return "NO "
        else:
            return "PCOS"
    elif clf_name == "Catboost Classifier":
        prediction = loaded_model_cat.predict(input_data_reshaped)
        if prediction == 0:
            return "NO PCOS"
        else:
            return "PCOS"


def main():

    # Title for web page
    st.title('PCOS Prediction using Machine Learning')
    if clf_name == 'Logistic Regression':
        st.write("Logistic Regression")
    elif clf_name == 'XGRBF Classifier':
        st.write("XGRBF Classifier")
    elif clf_name == 'Catboost Classifier':
        st.write("Catboost Classifier")
    # Getting inputs from the user
    age = st.number_input('Please Mention your Age')
    stresslevels = st.number_input(
        'Give your Stress levels ranging from 1 to 5(1 => Lowest, 5 => Highest)')
    sleepduration = st.number_input('How many hours do you sleep at night ? :')
    sleeprating = st.number_input(
        'Rate your daily sleep from 1 to 5(1 => bad, 5 => Good)')
    smoking = st.number_input(
        'Do you Smoke? (If yes mention 1 and if No mention 0) :')
    alcohol = st.number_input(
        'Do you drink Alcohol? (If yes mention 1 and if No mention 0) :')
    delayInPeriods = st.number_input("Delay in Periods (In Days): ")
    emotionalStatus = st.number_input("Emotional Status (answer 1 or 0): ")
    MalePattern_you = st.number_input(
        "Signs of Male Pattern(If yes then write 1 or else write 0): ")
    pms = st.number_input(
        "Premenstrual Syndrome ? (if yes => 1 or else => 0): ")

    # Code for prediction
    diagnosis = ''

    # Creating a button for prediction
    if st.button("PCOS Test Result"):
        diagnosis = pcos_prediction([age,
                                     stresslevels,
                                     sleepduration,
                                     sleeprating,
                                     smoking,
                                     alcohol,
                                     delayInPeriods,
                                     emotionalStatus,
                                     MalePattern_you,
                                     pms])
    st.success(diagnosis)


if __name__ == '__main__':
    main()
