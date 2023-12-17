import pickle
import tensorflow as tf
from tensorflow import keras
import numpy as np
from music21 import instrument, note, stream, chord
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, BatchNormalization as BatchNorm, Activation

def generate_music():
    with open('lstm/data/notes', 'rb') as file_path:
        musical_data = pickle.load(file_path)

    pitch_names = sorted(set(item for item in musical_data))
    
    unique_elements = len(set(musical_data))

    network_input, normalized_input = prepare_sequences(musical_data, pitch_names, unique_elements)
    model = create_model(normalized_input, unique_elements)
    prediction_output = generate_new_music(model, network_input, pitch_names, unique_elements)
    save_mid_file(prediction_output)

def prepare_sequences(musical_data, pitch_names, unique_elements):
    element_to_int = dict((element, number) for number, element in enumerate(pitch_names))

    sequence_length = 100
    network_input = []
    output = []
    for i in range(0, len(musical_data) - sequence_length, 1):
        sequence_in = musical_data[i:i + sequence_length]
        sequence_out = musical_data[i + sequence_length]
        network_input.append([element_to_int[char] for char in sequence_in])
        output.append(element_to_int[sequence_out])

    num_patterns = len(network_input)

    normalized_input = np.reshape(network_input, (num_patterns, sequence_length, 1))
    normalized_input = normalized_input / float(unique_elements)

    return (network_input, normalized_input)

def create_model(network_input, unique_elements):
    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        recurrent_dropout=0.3,
        return_sequences=True
    ))
    model.add(LSTM(512, return_sequences=True, recurrent_dropout=0.3,))
    model.add(LSTM(512))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(unique_elements))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    model.load_weights('lstm/data/trained_weights.hdf5')

    return model

def generate_new_music(model, network_input, pitch_names, unique_elements):

    start = np.random.randint(0, len(network_input)-1)

    int_to_element = dict((number, element) for number, element in enumerate(pitch_names))

    pattern = network_input[start]
    prediction_output = []

    for element_index in range(500):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(unique_elements)

        prediction = model.predict(prediction_input, verbose=0)

        index = np.argmax(prediction)
        result = int_to_element[index]
        prediction_output.append(result)

        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    return prediction_output

def save_mid_file(prediction_output):
    offset = 0
    output_elements = []

    for pattern in prediction_output:
        if ('.' in pattern) or pattern.isdigit():
            elements_in_chord = pattern.split('.')
            elements = []
            for current_element in elements_in_chord:
                new_element = note.Note(int(current_element))
                new_element.storedInstrument = instrument.Piano()
                elements.append(new_element)
            new_chord = chord.Chord(elements)
            new_chord.offset = offset
            output_elements.append(new_chord)
        else:
            new_element = note.Note(pattern)
            new_element.offset = offset
            new_element.storedInstrument = instrument.Piano()
            output_elements.append(new_element)

        offset += 0.5

    midi_stream = stream.Stream(output_elements)
    midi_stream.write('midi', fp='generated_music.mid')

if __name__ == '__main__':
    generate_music()
