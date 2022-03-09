import numpy as np
import pickle
import streamlit as st

# Loading the saved model

loaded_model = pickle.load(open('trained_model.sav', 'rb'))

# creating a function for prediction


def pcos_prediction(input_data):

    input_data_as_numpy_array = np.asarray(input_data)

    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)
    # cat_clf.predict(input_data_reshaped)
    print(prediction)
    if prediction == 0:
        return "NO PCOS"
    else:
        return "PCOS"


def main():

    # Title for web page
    st.title('PCOS Prediction using Machine Learning Web App')

    # Getting inputs from the user
    age = st.text_input('Please Mention your Age')
    stresslevels = st.text_input(
        'Give your Stress levels ranging from 1 to 5(1 => Lowest, 5 => Highest)')
    sleepduration = st.text_input('How many hours do you sleep at night ? :')
    sleeprating = st.text_input(
        'Rate your daily sleep from 1 to 5(1 => bad, 5 => Good)')
    smoking = st.text_input(
        'Do you Smoke? (If yes mention 1 and if No mention 0) :')
    alcohol = st.text_input(
        'Do you drink Alcohol? (If yes mention 1 and if No mention 0) :')
    timebetweenperiods = st.text_input("Time between your periods: ")
    delayInPeriods = st.text_input("Delay in Periods: ")
    emotionalStatus = st.text_input("Emotional Status: ")
    MalePattern_you = st.text_input(
        "Signs of Male Pattern(If yes then write 1 or else write 0): ")
    pms = st.text_input("Premenstrual Syndrome ? : ")

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
                                     timebetweenperiods,
                                     delayInPeriods,
                                     emotionalStatus,
                                     MalePattern_you,
                                     pms])
    st.success(diagnosis)


if __name__ == '__main__':
    main()
