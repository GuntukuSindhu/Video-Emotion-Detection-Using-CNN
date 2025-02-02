import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, accuracy_score

# Emotion dictionary
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Load JSON and create model
model_json_path = 'C:\\Users\\sindh\\OneDrive\\Desktop\\mini project\\model\\emotion_model.json'
model_weights_path = 'C:\\Users\\sindh\\OneDrive\\Desktop\\mini project\\model\\emotion_model.h5'

# Load model from JSON
try:
    with open(model_json_path, 'r') as json_file:
        loaded_model_json = json_file.read()
except FileNotFoundError:
    print("The model JSON file does not exist.")
    exit()
except Exception as e:
    print("An error occurred while reading the model JSON file:", str(e))
    exit()

# Load model from JSON
try:
    emotion_model = model_from_json(loaded_model_json)
except Exception as e:
    print("Failed to create model from JSON:", str(e))
    exit()

# Load weights into the model
try:
    emotion_model.load_weights(model_weights_path)
except Exception as e:
    print("Failed to load model weights:", str(e))
    exit()

print("Loaded model from disk")

# Initialize image data generator with rescaling
test_data_gen = ImageDataGenerator(rescale=1./255)

# Specify the path to the test directory
test_dir = r'C:\\Users\\sindh\\OneDrive\\Desktop\\mini project\\test'

# Check if the test directory exists
if not os.path.exists(test_dir):
    print("The test directory does not exist.")
    exit()

# Preprocess all test images
test_generator = test_data_gen.flow_from_directory(
    test_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical',
    shuffle=False  # Important to keep the order for accurate predictions
)

# Ensure there are images to predict
if test_generator.samples == 0:
    print("No images found in the test directory.")
    exit()

# Do prediction on test data
try:
    predictions = emotion_model.predict(test_generator)
except Exception as e:
    print("Failed to make predictions:", str(e))
    exit()

# Confusion matrix
predicted_labels = np.argmax(predictions, axis=1)
c_matrix = confusion_matrix(test_generator.classes, predicted_labels)  
print("Confusion Matrix:\n", c_matrix)

# Display confusion matrix
cm_display = ConfusionMatrixDisplay(confusion_matrix=c_matrix, display_labels=list(emotion_dict.values()))
cm_display.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# Classification report
print("-----------------------------------------------------------------")
print(classification_report(test_generator.classes, predicted_labels, target_names=list(emotion_dict.values())))

# Calculate accuracy
accuracy = accuracy_score(test_generator.classes, predicted_labels)
print(f"Accuracy: {accuracy * 100:.2f}%")