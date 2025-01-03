from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
import numpy as np

# Load VGG16 model
model = VGG16()
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

# Extract features from an image
def extract_features(image):
    """Extract features from an image using VGG16."""
    features = model.predict(image)
    return np.squeeze(features)

if __name__ == "__main__":
    print("Feature extraction module loaded.")
