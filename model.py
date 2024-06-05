import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Embedding, LSTM, add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
#Encoder Part of the Model
inputs1 = Input(shape=(4096,), name="image")
features1 = Dropout(0.4)(inputs1)
features2 = Dense(256, activation='relu')(features1)
#For Text Captions
inputs2 = Input(shape=(Max_length,), name="text")
sequences1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
sequences2 = Dropout(0.4)(sequences1)
sequences3 = LSTM(256)(sequences2)

#Decoder Part of the Model
decoder1 = add([features2, sequences3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)

model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam')
