from keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from keras.models import Model

def Img_to_Cap_modelmodel(vocab_size, max_length):
    """Define the image captioning model."""
    # Feature extractor model input
    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    # Sequence model input
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    # Decoder model
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    # Merge the models
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    return model

if __name__ == "__main__":
    print("Model module loaded.")
