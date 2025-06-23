import tensorflow as tf
import shap
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
"""def load_data():
    X,Y=[],[]
    for i in range(1,5):
        with open(f'processed_hand_data_{i}.pkl', 'rb') as f:
            data = pickle.load(f)
            X.append(data)
            labels = [i-1] * (data.shape[0])
            Y.extend(labels)
    X = np.concatenate(X, axis=0)
    Y = np.array(Y)
    return X,Y.reshape(-1,1)
X, Y = load_data()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
def create_model(input_shape):
    model = models.Sequential([
        layers.Flatten(input_shape=input_shape),
        layers.Dense(32, activation='relu'),
        layers.Dense(4 , activation='softmax')
    ])
    return model
model = create_model((21,3))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=30, batch_size=32, validation_split=0.2)
model.evaluate(X_test, Y_test)
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(Y_test, predicted_classes))
predictions = model.predict(X_train)
predicted_classes = np.argmax(predictions, axis=1)
print(confusion_matrix(Y_train, predicted_classes))
background = X_train[:100]  # Use 100 samples as background

# Create SHAP explainer
explainer = shap.DeepExplainer(model, background)

# Get SHAP values for test samples
test_samples = X_test[:10]  # Analyze 10 test samples
shap_values = explainer.shap_values(test_samples)

# Visualize results
shap.image_plot(shap_values, test_samples)
model.save('hand_gesture_model.h5')"""


# Method 2: From Keras model
model = tf.keras.models.load_model('hand_gesture_model.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the converted model
with open('hand_gesture_model_lite.tflite', 'wb') as f:
    f.write(tflite_model)