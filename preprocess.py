import os
import pickle
import numpy as np
from tqdm.notebook import tqdm
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
# Load and preprocess images
def preprocess_image(image_path):
    """Load and preprocess an image for feature extraction."""
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image

# Tokenization for text
def create_tokenizer(captions):
    """Create a tokenizer from a list of captions."""
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(captions)
    return tokenizer

# Sequence padding
def pad_text_sequences(sequences, max_length):
    """Pad sequences to the same length."""
    return pad_sequences(sequences, maxlen=max_length, padding='post')

if __name__ == "__main__":
    print("Preprocessing module loaded.")
