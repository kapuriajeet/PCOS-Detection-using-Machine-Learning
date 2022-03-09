import numpy as np
import pickle

# Loading the saved model

loaded_model = pickle.load(open('trained_model.sav', 'rb'))

input_data = (21, 3, 7, 4, 0, 0, 1, 2, 0, 0, 1)

input_data_as_numpy_array = np.asarray(input_data)

input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = loaded_model.predict(input_data_reshaped)
# cat_clf.predict(input_data_reshaped)
print(prediction)
if prediction == 0:
    print("NO PCOS")
else:
    print("PCOS")
