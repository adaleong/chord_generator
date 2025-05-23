# Documentation

# About
My project is a chord generation project for musicians who want a starting point. It will generate text based chords. There are two parts in this documentation as I created it in two phases.

# Research
I started with this dataset from Kaggle: https://www.kaggle.com/datasets/henryshan/a-dataset-of-666000-chord-progressions?resource=download In understanding the data, I found that unfortunately, this dataset was huge and inconsistent and had too much information such as artist ID's, genres, etc. The data from the chords itself were inconsistent as well.

I also found this paper: https://paperswithcode.com/paper/jambot-music-theory-aware-chord-based - a music theory aware chord generation of polyphonic music with LSTMs.

# Troubleshooting
With the dataset being so big, I decided to put it into chatgpt to ask it to analyze the data for me. Unfortunately, it was too big of a file and even after simplying the code to one column, having over 600k coloums just wasn't a realistic number to sort through by myself. I also tried instead of asking it to analyze it, to just use it as training data but that also didn't work due to the file size.

So I decided to create a music theory aware model instead.

# Demo time!

# Classification for Major, Minor, and 7th Chords
I wanted to first be able to classify the chords between major, minor, and 7th chords. I used ChatGPT in this part with the prompt being "create python code to classify major, minor, and seventh chords, import a classifier."

1. Importing libraries
```
from sklearn.ensemble import RandomForestClassifier
import numpy as np
```
- RandomForestClassifier from scikit-learn is used for classification and numpy (np) is a library for numerical data

2. Defining chord structures:
```
CHORDS = {
    "Major": [0, 4, 7],
    "Minor": [0, 3, 7],
    "Seventh": [0, 4, 7, 10],
}
```
- This maps chord names to intervals from the root note. For example, major chord is root (0), major 3rd (4), and perfect fifth (7). The numbers represent the numbers of semitones between each of the notes.

3. Finding the maximum chord length
```
max_length = max(len(chord) for chord in CHORDS.values())
```
- This determines the maximum number of notes in any chord. For major and minor chords, it is three and for 7th chords, it's 4 notes.

4. Function to pad chords
```
def pad_chord(chord, length, pad_value=-1):
    return chord + [pad_value] * (length - len(chord))
```
- This ensures that all of the chords have the same length by filling missing positions with -1. So for major chords that are not major 7th chords, [0, 4, 7] turns into [0, 4, 7, -1].

5. Preparing training data.
```
X_train = []
y_train = []

for label, intervals in CHORDS.items():
    X_train.append(pad_chord(intervals, max_length))
    y_train.append(label)
```
- This loops through the chords directory and pads each chord to the maximum length, while storing the chord structure x_train (a list of lists where the chords are represented by the string of numbers like [0, 4, 7, -1]) and the label y_train which contains the labels (major, minor, seventh)

6. Convert to Numpy Arrays
```
X_train = np.array(X_train)
y_train = np.array(y_train)
```
- This converts lists into NumPy arrays for efficiency when training.

7. Training the classifier
```
clf = RandomForestClassifier(n_estimators=10, random_state=42)
clf.fit(X_train, y_train)
```
- Creates a randomforestclassifier, `n_estimators=10` uses 10 decision trees and `random_state=42` ensures reproducibility. It is being trained on the x_train chord data and y_train labels.

8. Classifying the chords
```
def classify_chord(notes):
    """Classifies a chord given its notes as a list of semitone intervals from the root."""
    notes = sorted(notes)  # Ensure proper order
    padded_notes = pad_chord(notes, max_length)  # Ensure consistent input size
    prediction = clf.predict([padded_notes])[0]
    return prediction
```
- This takes a list of the semitone intervals as input, sorts the notes to ensure proper order, pads the chord to match training format, and predicts the chord type using the trained model clf.predict()

9. Testing with example:
For this example, we're testing C Major:
```
example_chord = [0, 4, 7]  # C Major
print("Chord classification:", classify_chord(example_chord))
```
- It classifies [0, 4, 7] to `classify_chord()` and the output is Major.

# Chords beyond chords in C Root Position
Right now, the classifier only recognizes chords in root position, and I wanted it to be bale to recognize chords in any key. So instead of using absolute values, shifting all the notes so that the root note is always 0. For example, D Major would be (2, 6, 9) but it gets turned into (0, 4, 7)

The next step is then to train transposed versions of chords to generate training data in all 12 keys.

# Part Two
The updated code generates a string of 6 chords based on the root note I give it. To start, I imported a random module `import random`.

I also created a list of note names:
```
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
```
These note names correspond with the numbers 0-11 (as the numbers that make up the chords).

I wanted it so that the chords each had at least one note in common with each other so the chords would actually sound good together, so:
```
def generate_random_chords_with_overlap(root_note, num_chords=6):
    random_chords = []
    previous_chord_notes = None
    root_note_name = NOTE_NAMES[root_note]  # Convert root note to its name
```
`generate_random_chords_with_overlap:` is a function that generates a series of random chords starting from a given root note labeled as`(root_note)`, ensuring that each chord has at least one note in common with the previous chord. The number of chords to generate is specified by `num_chords` (default 6).

The number of chords being generated as well as the root note can both be edited.


```
chord_root_name = NOTE_NAMES[chord_notes[0] % 12]  # Mod 12 to get the actual note in the 0-11 range
random_chords.append((chord_root_name, chord_type, chord_notes))
previous_chord_notes = chord_notes
```

`chord_root_name:` The root note of the chord is extracted by taking the first note in the chord. Mod 12 is used to make sure that it stays within the values since there are only 12 notes. It "wraps" numbers around so that values greater than 11 or less than 0 are correctly mapped back into the 0-11 range (C to B). So 13 would be 1 for example.

The chord's root name, type, and notes are stored in `random_chords`, and `previous_chord_notes` is updated for the next loop since it has to generate 6 times.

The line `return random_chords
` returns the list of randomly generated chords. 

```
root_note = 0  # C major scale
random_chords = generate_random_chords_with_overlap(root_note, num_chords=6)
```
This sets the root note, you can set it to any between 0-11 and the next line selects the number of chords to generate. I chose 6 as it is a standard phrasing.

Last step:
```
for chord_root_name, chord_type, chord_notes in random_chords:
    chord_class = classify_chord(chord_notes)
    print(f"Chord: {chord_root_name} {chord_type} | Notes: {chord_notes} | Type: {chord_class}")
```
This loop goes through each generated chord, classifies it using `classify_chord`, and prints the chord details: root name, type, notes, and predicted chord type.

# Accomplishments and Future Work
I was able to create a random chord generator, it is music theory aware in the way where the chords going to each other makes sense.

However, I wasn't able to use the dataset and there is definitely more music theory I can add to this, and only three types of chords are being generated. In the future, I want it to not only use the music theory training but also the dataset. I also want it to generate audio so that we can listen to the chords.

# Documentation Part Two

# Introduction
This is a chord generation model that generates random chord progressions based on predefined chord structures, applies smooth voice leading between chords, and ensures a final resolution to tonic (I) after the dominant (V7). The generated chords can be played back as MIDI files, with customization for key types, number of chords, and tempo.

## 1. Initial Chord Structures
- Defined the chord structures with interval patterns for various chord types such as Major, Minor, 7th chords, extended chords like 9th, 11th, and 13th, and diminished chords.
```
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
```
## 2. Avoiding root notes next to each other (tritones)
- The current model generated notes next to each other making it sound bad
- introduced a constraint to ensure that root notes are not one semitone (tritone) apart:
```
# Function to check if the root note is a tritone away from another root note
def is_tritone_away(root1, root2):
    return abs(root1 - root2) == 6  # 6 semitones = tritone
```

## 3. Repeated root notes
- Repeating root notes resulted in less variety
- Updated the chord generation logic to make sure that root notes differ by more than just one semitone from each other, avoiding immediate repetition of the root.
```
root = start_note if i == 0 else random.choice([n for n in range(12) if not is_tritone_away(n, start_note)])
```
This ensures that root notes are not immediately repeated or placed next to each other.

## 4. Recurring Themes and Smooth Voice Leading
I wanted to add recurring themes to make the progression feel more cohesive and transition smoothly. I wanted to minimize large jumps between notes.
- Added a logic to generate a theme (a sequence of 4 chords) and repeated it throughout the progression.
```
# Pattern for recurring themes (reused motif)
theme = random.sample(possible_chords, 4)  # A short theme to repeat
```
For a more coherent and thematic progression:
```
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
```
## 5. Final Resolution
- Created a (V-1) resolution so that the final chord was always the tonic chord (the first chord)
```
# Ensure the last two chords end in resolution
if i == num_chords - 2:
    chord_type = "Dominant7"  # Dominant7 (V7) chord for second last
elif i == num_chords - 1:
    chord_type = "Major"  # Major (I) chord for the last chord
```

## 6. Adjustments / smoother transitions
The final generated progression was playable as a MIDI file using the music21 library, and we added functionality for adjusting the tempo (BPM) of the progression.

## 7. Using music21 for MIDI playback
- Used the music21 library to convert the generated chords into MIDI format and to play the resulting music
- Originally, I tried to use homebrew but something kept going wrong in the installation process where it wasn't registering that it was installed in my computer, so I switched to music21 upon recommendation.
```
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
```
1. The chord to stream converts the generated chord progression into a music21 stream object
2. The tempo sets the tempo for playback (quarter note = x bpm)
3. MIDI playback: using midi.midifile to export the midi stream for playback
At first when I was doing this, the MIDI was playing really fast so I defined the bpm. However, this still doesn't work and I can't figure out why.
```
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
```
^ This generates the chords and plays it back at a specified BPM

## 8. Acessible "interface"
```
generate_and_play_chords(start_note=0, key_type="Major", num_chords=8, bpm=80, length=1)
```
Put everything customizable in the same spot for ease of access.

## 9. Range of Notes
- With the range, it made the notes super low since it the first note started at 0.
- added a function constrain_notes_to_range() that adjusts all pitches to keep the average within a playable range:
```
def constrain_notes_to_range(notes, low=60, high=72):
    avg = sum(notes) / len(notes)
    target_avg = (low + high) / 2
    shift = round(target_avg - avg)
    return [note + shift for note in notes]
```

## 10. Dataset
- I was not able to implement the dataset I made, the input format was too ambigous since I needed the raw MIDI note numbers, pitch names, roman numeral analysis, and the chord symbols
- There were too many variables I would have to implement into the dataset

## 11. Chord Extensions
- Added chord extensions (9ths, 7ths, 11ths, and 13ths)
In addition to the triad format, I added extensions to those. 
```
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
```
Diminished, half diminished, and dominant chords were also added.

## 12. Chord Mapping / Labeling
Chord mapping for consistent labeling:
```
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
```

## 13. Inversions and Voicings
This also helped with some of the weird jumps, but having consistent root note voicings. Inversions reorder a chord’s notes so that a note other than the root appears in the bass (lowest position). This was implemented to avoid repetitive "block" voicings and to support voice leading between chords.
```
def apply_random_inversion(notes):
    num_inversions = random.randint(0, len(notes) - 1)
    inverted = notes[num_inversions:] + [n + 12 for n in notes[:num_inversions]]
    return sorted(inverted)
```
This Selects a random inversion up to the number of chord tones, rotates chord tones and raises earlier notes by an octave (i.e., adds 12 semitones), and maintains consistent pitch structure while altering the bass note.

Voicing determines the spacing and distribution of chord tones across the pitch range. To keep voicings natural and playable. This normalizes voicings within a reasonable MIDI pitch range (default: middle C to upper octave), ensures the average pitch of the chord is centered within the target range, and avoids overly low or high clusters.
```
def constrain_notes_to_range(notes, low=60, high=72):
    avg = sum(notes) / len(notes)
    target_avg = (low + high) / 2
    shift = round(target_avg - avg)
    return [note + shift for note in notes]
```

## 14. Voice Leading
Smooth transitions between chords were prioritized through voice leading adjustments, minimizing large jumps:
```
def smooth_voice_leading(current_notes, next_notes):
    smooth_notes = []
    for current, next_ in zip(current_notes, next_notes):
        smooth_notes.append(next_ if current != next_ else current)
    return smooth_notes
```
^ makes it sound less awkward

## 15. Chord label accuracy with inversions
With the changes to inversions and voicings, things can get a little weird, so inversions are labeled using slash notation such as (Dmaj7/#F)

## 16. Fuzzy Chord System
To address the variability in voicings (e.g., inversions, dropped notes, extra tones), a fuzzy matching algorithm was created for flexible classification. It helps with recognizing chords with extensions or chords that might fit different labels because of those extensions.
```
def classify_chord_fuzzy(notes):
    # Match input notes (normalized to pitch class) against all known chords
    # Score by shared pitch classes and return the best match