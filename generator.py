from sklearn.ensemble import RandomForestClassifier
import numpy as np
import random
from music21 import stream, note, chord, midi, tempo

# --- Define chord structures (with explicit quality) ---
CHORDS = {
    "Major": [0, 4, 7],
    "Minor": [0, 3, 7],
    "Major6": [0, 4, 7, 9],
    "Minor6": [0, 3, 7, 9],
    "Dominant7": [0, 4, 7, 10],
    "Major7": [0, 4, 7, 11],
    "Minor7": [0, 3, 7, 10],
    "Major9": [0, 4, 7, 11, 14],
    "Minor9": [0, 3, 7, 10, 14],
    "Dominant9": [0, 4, 7, 10, 14],
    "Major11": [0, 4, 7, 11, 14, 17],
    "Minor11": [0, 3, 7, 10, 14, 17],
    "Dominant11": [0, 4, 7, 10, 14, 17],
    "Major13": [0, 4, 7, 11, 14, 17, 21],
    "Minor13": [0, 3, 7, 10, 14, 17, 21],
    "Dominant13": [0, 4, 7, 10, 14, 17, 21],
    "Diminished": [0, 3, 6],
    "Diminished7": [0, 3, 6, 9],
    "HalfDiminished7": [0, 3, 6, 10],
}

CHORD_LABELS = {
    "Major": "maj",
    "Minor": "min",
    "Major6": "maj6",
    "Minor6": "min6",
    "Dominant7": "7",
    "Major7": "maj7",
    "Minor7": "min7",
    "Major9": "maj9",
    "Minor9": "min9",
    "Dominant9": "9",
    "Major11": "maj11",
    "Minor11": "min11",
    "Dominant11": "11",
    "Major13": "maj13",
    "Minor13": "min13",
    "Dominant13": "13",
    "Diminished": "dim",
    "Diminished7": "dim7",
    "HalfDiminished7": "min7b5",
}

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

max_length = max(len(c) for c in CHORDS.values())

def pad_chord(chord, length, pad_value=-1):
    return chord + [pad_value] * (length - len(chord))

X_train = []
y_train = []

for label, intervals in CHORDS.items():
    X_train.append(pad_chord(intervals, max_length))
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

clf = RandomForestClassifier(n_estimators=10, random_state=42)
clf.fit(X_train, y_train)

def classify_chord_fuzzy(notes):
    notes = sorted(notes)
    max_score = -1
    best_match = None

    for chord_name, intervals in CHORDS.items():
        input_root = notes[0] % 12
        shifted_intervals = [(input_root + interval) % 12 for interval in intervals]
        input_notes_set = set(n % 12 for n in notes)
        chord_notes_set = set(shifted_intervals)
        score = len(input_notes_set.intersection(chord_notes_set))
        if score > max_score:
            max_score = score
            best_match = chord_name

    return best_match

def apply_random_inversion(notes):
    num_inversions = random.randint(0, len(notes) - 1)
    inverted = notes[num_inversions:] + [n + 12 for n in notes[:num_inversions]]
    return sorted(inverted)

def get_bass_note_label(notes):
    return NOTE_NAMES[notes[0] % 12]

def constrain_notes_to_range(notes, low=60, high=72):
    avg = sum(notes) / len(notes)
    target_avg = (low + high) / 2
    shift = round(target_avg - avg)
    return [note + shift for note in notes]

def smooth_voice_leading(current_notes, next_notes):
    """Ensure smooth voice leading by minimizing large jumps."""
    smooth_notes = []
    for current, next_ in zip(current_notes, next_notes):
        if current == next_:
            smooth_notes.append(current)
        else:
            # Move the note to the closest target note
            smooth_notes.append(next_)
    return smooth_notes

def generate_random_chords(start_note=0, key_type="Major", num_chords=6):
    """Generate a random progression with the desired number of chords."""
    possible_chords = [k for k in CHORDS if k.startswith(key_type)]
    random_chords = []
    previous_notes = None

    # Pattern for recurring themes (we can reuse motifs)
    theme = random.sample(possible_chords, 4)  # A short theme to repeat

    for i in range(num_chords - 2):  # Leave room for the last two chords (V-I resolution)
        # Avoid root notes that are a tritone (half-step or 6 semitones apart)
        while True:
            chord_type = theme[i % 4]  # Repeating a motif (can vary its position or inversion)
            intervals = CHORDS[chord_type]
            
            # Ensure the root notes are not adjacent (no half-step intervals between them)
            root = start_note if i == 0 else random.choice([n for n in range(12) if abs(n - start_note) > 1])
            notes = [root + interval for interval in intervals]
            notes = apply_random_inversion(notes)

            if previous_notes:
                notes = smooth_voice_leading(previous_notes, notes)

            notes = constrain_notes_to_range(notes)
            if abs(notes[0] - start_note) > 1:  # Ensure the root is not adjacent
                break

        root_label = NOTE_NAMES[notes[0] % 12]
        chord_label = CHORD_LABELS[chord_type]
        bass = get_bass_note_label(notes)

        full_label = f"{root_label}{chord_label}"
        if bass != root_label:
            full_label += f"/{bass}"

        random_chords.append((full_label, chord_type, notes))
        previous_notes = notes

    # Now ensure the last two chords are V-I or IV-I resolution
    tonic_chord = "Major" if key_type == "Major" else "Minor"
    dominant_chord = "Dominant7"
    
    # Resolve with a dominant chord (V7) followed by tonic chord (I)
    dominant_root = random.choice([n for n in range(12)])
    tonic_root = dominant_root + 5  # Perfect fifth above dominant
    
    dominant_notes = [dominant_root + interval for interval in CHORDS[dominant_chord]]
    tonic_notes = [tonic_root + interval for interval in CHORDS[tonic_chord]]

    # Apply inversion and smooth voice leading if necessary
    dominant_notes = apply_random_inversion(dominant_notes)
    tonic_notes = apply_random_inversion(tonic_notes)
    dominant_notes = constrain_notes_to_range(dominant_notes)
    tonic_notes = constrain_notes_to_range(tonic_notes)

    random_chords.append(("V7", dominant_chord, dominant_notes))
    random_chords.append(("I", tonic_chord, tonic_notes))

    return random_chords

def chords_to_stream(chords, bpm=60):
    s = stream.Stream()
    p = stream.Part()
    p.append(tempo.MetronomeMark(number=bpm))  # Attach tempo to the Part

    for _, _, notes in chords:
        c = chord.Chord(notes)
        c.duration.quarterLength = 1
        p.append(c)

    s.append(p)
    return s

# --- Function to Generate and Play Chords ---
def generate_and_play_chords(start_note=0, key_type="Major", num_chords=6, bpm=60, length=1):
    chords = generate_random_chords(
        start_note=start_note,
        key_type=key_type,
        num_chords=num_chords
    )

    print("\nGenerated Chords:")
    for label, chord_type, notes in chords:
        fuzzy_type = classify_chord_fuzzy(notes)
        print(f"Chord: {label} | Notes: {notes} | Fuzzy Type: {fuzzy_type}")

    midi_stream = chords_to_stream(chords, bpm=bpm)
    midi_stream.show('midi')  # This will display the midi file and play it in Jupyter

# Example usage
# You can change the following parameters:
# - start_note: The starting note (0 = C, 1 = C#, 2 = D, etc.)
# - key_type: "Major" or "Minor"
# - num_chords: How many chords in the progression
# - bpm: Tempo (beats per minute)
# - length: Length of each chord in quarter notes

generate_and_play_chords(start_note=5, key_type="Major", num_chords=16, bpm=30, length=1)
