import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Embedding, LSTM, add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

class MyModel(tf.keras.Model):
    def __init__(self, max_length, vocab_size):
        super(MyModel, self).__init__()

        self.inputs1 = Input(shape=(4096,))
        self.fe1 = Dropout(0.4)
        self.fe2 = Dense(256, activation='relu')

        self.inputs2 = Input(shape=(max_length,))
        self.se1 = Embedding(vocab_size, 256, mask_zero=True)
        self.se2 = Dropout(0.4)
        self.se3 = LSTM(256)

        self.decoder1 = add
        self.decoder2 = Dense(256, activation='relu')
        self.outputs = Dense(vocab_size, activation='softmax')

    @tf.function
    def call(self, inputs1, inputs2):
        x1 = self.fe1(inputs1)
        x1 = self.fe2(x1)

        x2 = self.se1(inputs2)
        x2 = self.se2(x2)
        x2 = self.se3(x2)

        x = self.decoder1([x1, x2])
        x = self.decoder2(x)
        outputs = self.outputs(x)

        return outputs

# Example usage
max_length = 100
vocab_size = 10000
model = MyModel(max_length, vocab_size)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=Adam())
