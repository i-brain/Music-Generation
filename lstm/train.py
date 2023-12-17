from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np

import os
import pretty_midi
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, BatchNormalization, Dropout
import numpy as np
import os
import pretty_midi
from keras.utils import to_categorical


midi_folder = 'midi_songs'

def process_midi_file(file_path):
    midi_data = pretty_midi.PrettyMIDI(file_path)
    
    features = [note.start for instrument in midi_data.instruments for note in instrument.notes]
    
    return features

all_features = []
for filename in os.listdir(midi_folder):
    if filename.endswith(".mid"):
        file_path = os.path.join(midi_folder, filename)
        features = process_midi_file(file_path)
        all_features.extend(features)

sequence_length = 100  
num_features = 1  

X_train = []
y_train = []

for i in range(len(all_features) - sequence_length):
    sequence_in = all_features[i:i + sequence_length]
    sequence_out = all_features[i + sequence_length]
    X_train.append(sequence_in)
    y_train.append(sequence_out)

X_train = np.array(X_train)
y_train = np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], sequence_length, num_features))

model = Sequential()
model.add(LSTM(512, input_shape=(sequence_length, num_features), return_sequences=True, recurrent_dropout=0.3))
model.add(LSTM(512, return_sequences=True, recurrent_dropout=0.3))
model.add(LSTM(512))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(num_features, activation='softmax'))  

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

model.fit(X_train, y_train, epochs=10, batch_size=64)

model.save_weights('data/trained_weights.hdf5')
