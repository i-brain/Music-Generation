import os
import random
from music21 import converter, note, chord, stream, pitch
from collections import defaultdict
from difflib import SequenceMatcher


def convert_music_to_notes(file_path):
    try:
        score = converter.parse(file_path)
    except Exception as e:
        print(f"Error loading the music file: {e}")
        return []

    notes = []
    for element in score.flatten().getElementsByClass(['Note', 'Chord']):
        if 'Chord' in element.classes:
            notes.append(element)
        elif 'Note' in element.classes:
            notes.append(element)

    return notes


def calculate_transition_probabilities(note_sequences):
    transition_counts = defaultdict(lambda: defaultdict(int))

    for note_sequence in note_sequences:
        for i in range(len(note_sequence) - 1):
            current_note = note_sequence[i]
            next_note = note_sequence[i + 1]

            current_note_str = note_str(current_note)
            next_note_str = note_str(next_note)

            transition_counts[current_note_str][next_note_str] += 1

    return transition_counts


def print_transition_probabilities(probabilities, total_notes):
    print(f"Total number of notes: {total_notes}")

    for current_note, next_notes in probabilities.items():
        total_count = sum(next_notes.values())


        for next_note, count in next_notes.items():
            percentage = (count / total_count) * 100


def note_str(note_obj):
    if isinstance(note_obj, chord.Chord):
        return ', '.join(str(p) for p in note_obj.pitches)
    elif isinstance(note_obj, note.Note):
        return note_obj.nameWithOctave
    else:
        return str(note_obj)


def calculate_melodic_contour(notes):
    contour = []

    for i in range(len(notes) - 1):
        current_note = notes[i]
        next_note = notes[i + 1]

        if isinstance(current_note, str):
            current_pitches = [pitch.Pitch(p) for p in current_note.split(',')]
            current_pitch = current_pitches[0].ps
        elif isinstance(current_note, chord.Chord):
            current_pitch = current_note.pitches[0].ps
        else:
            current_pitch = current_note.pitch.ps

        if isinstance(next_note, str):
            next_pitches = [pitch.Pitch(p) for p in next_note.split(',')]
            next_pitch = next_pitches[0].ps
        elif isinstance(next_note, chord.Chord):
            next_pitch = next_note.pitches[0].ps
        else:
            next_pitch = next_note.pitch.ps

        if current_pitch < next_pitch:
            contour.append(1)  
        elif current_pitch > next_pitch:
            contour.append(-1)  
        else:
            contour.append(0)  

    return contour


def select_notes(transition_counts, start_note, target_num_notes, max_consecutive_occurrences=3):
    selected_notes = [start_note]
    current_note = start_note
    consecutive_count = 0

    while len(selected_notes) < target_num_notes and consecutive_count < max_consecutive_occurrences:
        next_notes = transition_counts.get(current_note, {})
        next_notes = {note: count for note, count in next_notes.items() if note not in selected_notes}

        if next_notes:
            next_note = max(next_notes, key=next_notes.get)

            if next_note == current_note:
                consecutive_count += 1
            else:
                consecutive_count = 0

            selected_notes.append(next_note)
            current_note = next_note
        else:
            possible_notes = list(set(note for notes in transition_counts.values() for note in notes.keys()) - set(
                selected_notes))
            if possible_notes:
                random_note = random.choice(possible_notes)
                selected_notes.append(random_note)
                current_note = random_note
                consecutive_count = 0
            else:
                break

    return selected_notes[:target_num_notes]


def save_selected_notes(selected_notes, file_path):
    print(f"Generated Music Notes:")
    for note_str in selected_notes:
        print(note_str)

    selected_notes_stream = stream.Stream()

    for selected_note_str in selected_notes:
        if ',' in selected_note_str:
            pitches = [pitch.Pitch(p) for p in selected_note_str.split(',')]
            selected_chord = chord.Chord(pitches)
            selected_notes_stream.append(selected_chord)
        else:
            selected_note = note.Note(selected_note_str)
            selected_notes_stream.append(selected_note)

    selected_notes_stream.write('midi', fp=file_path)
    print(f"Selected MIDI file saved as: {file_path}")


def main():
    music_folder_path = 'music'
    all_notes = []

    for filename in os.listdir(music_folder_path):
        if filename.endswith(".mid"):
            file_path = os.path.join(music_folder_path, filename)
            notes = convert_music_to_notes(file_path)
            all_notes.extend(notes)

    transition_counts = calculate_transition_probabilities([all_notes])
    print_transition_probabilities(transition_counts, len(all_notes))

    target_num_notes = len(all_notes) // len(os.listdir(music_folder_path))
    reference_notes = convert_music_to_notes(os.path.join(music_folder_path, os.listdir(music_folder_path)[0]))
    reference_contour = calculate_melodic_contour(reference_notes)

    similarity_ratio = 0.0
    iteration_count = 0

    most_used_notes = sorted(transition_counts.keys(), key=lambda x: sum(transition_counts[x].values()), reverse=True)

    while similarity_ratio < 0.5:
        iteration_count += 1

        current_most_used_note = most_used_notes[(iteration_count - 1) % len(most_used_notes)]
        selected_notes = select_notes(transition_counts, current_most_used_note, target_num_notes)
        generated_contour = calculate_melodic_contour(selected_notes)
        similarity_ratio = SequenceMatcher(None, generated_contour, reference_contour).ratio()

        if similarity_ratio >= 0.5:
            save_selected_notes(selected_notes, f'selected_output_{iteration_count}.mid')

        print(f"\nIteration {iteration_count}:")
        print(f"Most Used Note: {current_most_used_note}")
        print(f"Melodic Contour Similarity: {similarity_ratio:.2f}")

        transition_counts = calculate_transition_probabilities([selected_notes])


if __name__ == "__main__":
    main()
