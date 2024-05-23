import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import librosa

from keras import layers
from keras import models
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dense, Dropout, Flatten
from keras.regularizers import l2
from IPython import display
from IPython.display import Audio


# Set the seed value for experiment reproducibility.
seed = 20
tf.random.set_seed(seed)
np.random.seed(seed)

DATASET_PATH = 'data/Mitch Dataset'

data_dir = pathlib.Path(DATASET_PATH)
if not data_dir.exists():
    tf.keras.utils.get_file(
        'Mitch Dataset.zip',
        # Mediafire updates download keys, open the link and copy new download link if stops working
        # origin="https://download1514.mediafire.com/5hofuxy0eabgvHu3HCzn8Un5AJr--m9oy48SIzQ7xIoN-MsgsEcvsdXFIRraGhYmnhMwq2FtIE8KizxbJsC0i94RZubkZAdP3K2nrXHlYwCRXXIECTYC-OG7Tum9wFqaRZUDzL0-U_3Vf3HlL3IE-oooMwVPL3-dx_h8eMIleOw/jgsi62ik68nzw39/archive.zip",
        origin="https://www.dropbox.com/s/p736vokha3240e6/MDLib2.2.zip?dl=1",
        extract=True,
        cache_dir='.', cache_subdir='data')

commands = np.array(tf.io.gfile.listdir(str(data_dir)))
commands = commands[(commands != 'README.md') & (commands != '.DS_Store')]
print('Commands:', commands)

train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
    directory=data_dir,
    batch_size=64,
    validation_split=0.2,
    seed=0,
    output_sequence_length=44100,
    subset='both')

label_names = np.array(train_ds.class_names)
print()
print("label names:", label_names)

print(train_ds.element_spec)


def squeeze(audio, labels):
    audio = tf.squeeze(audio, axis=-1)
    return audio, labels


train_ds = train_ds.map(squeeze, tf.data.AUTOTUNE)
val_ds = val_ds.map(squeeze, tf.data.AUTOTUNE)

test_ds = val_ds.shard(num_shards=2, index=0)
val_ds = val_ds.shard(num_shards=2, index=1)

for example_audio, example_labels in train_ds.take(1):
    print(example_audio.shape)
    print(example_labels.shape)

label_names[[1, 1, 3, 0]]

plt.figure(figsize=(16, 10))
rows = 3
cols = 3
n = rows * cols
for i in range(n):
    plt.subplot(rows, cols, i + 1)
    audio_signal = example_audio[i]
    plt.plot(audio_signal)
    plt.title(label_names[example_labels[i]])
    plt.yticks(np.arange(-1.2, 1.2, 0.2))
    plt.ylim([-1.1, 1.1])


def get_mel_spectrogram(waveform, sample_rate=44100, num_mel_bins=64, lower_edge_hertz=125, upper_edge_hertz=7500):
    # Parameters for the STFT
    frame_length = 255
    frame_step = 128
    fft_length = 255

    # Compute STFT
    stft = tf.signal.stft(waveform, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length)
    spectrogram = tf.abs(stft)

    # Warp the linear-scale, magnitude spectrograms into the mel-scale.
    num_spectrogram_bins = stft.shape[-1]
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz, upper_edge_hertz)
    mel_spectrogram = tf.tensordot(
        spectrogram, linear_to_mel_weight_matrix, 1)
    mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate(
        linear_to_mel_weight_matrix.shape[-1:]))

    # Add a 'channels' dimension
    mel_spectrogram = mel_spectrogram[..., tf.newaxis]

    return mel_spectrogram



for i in range(3):
    label = label_names[example_labels[i]]
    waveform = example_audio[i]
    spectrogram = get_mel_spectrogram(waveform)

    print('Label:', label)
    print('Waveform shape:', waveform.shape)
    print('Spectrogram shape:', spectrogram.shape)
    print('Audio playback')
    display.display(display.Audio(waveform, rate=32000))


def plot_spectrogram(spectrogram, ax):
    if len(spectrogram.shape) > 2:
        assert len(spectrogram.shape) == 3
        spectrogram = np.squeeze(spectrogram, axis=-1)
    # Convert the frequencies to log scale and transpose, so that the time is
    # represented on the x-axis (columns).
    # Add an epsilon to avoid taking a log of zero.
    log_spec = np.log(spectrogram.T + np.finfo(float).eps)
    height = log_spec.shape[0]
    width = log_spec.shape[1]
    X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
    Y = range(height)
    ax.pcolormesh(X, Y, log_spec)


fig, axes = plt.subplots(2, figsize=(12, 8))
timescale = np.arange(waveform.shape[0])
axes[0].plot(timescale, waveform.numpy())
axes[0].set_title('Waveform')
axes[0].set_xlim([0, 32000])

plot_spectrogram(spectrogram.numpy(), axes[1])
axes[1].set_title('Spectrogram')
plt.suptitle(label.title())
# plt.show()


def make_spec_ds(ds, sample_rate=44100):
    return ds.map(
        map_func=lambda audio, label: (get_mel_spectrogram(audio, sample_rate=sample_rate), label),
        num_parallel_calls=tf.data.AUTOTUNE)



train_spectrogram_ds = make_spec_ds(train_ds)
val_spectrogram_ds = make_spec_ds(val_ds)
test_spectrogram_ds = make_spec_ds(test_ds)

for example_spectrograms, example_spect_labels in train_spectrogram_ds.take(1):
    break

rows = 3
cols = 3
n = rows * cols
fig, axes = plt.subplots(rows, cols, figsize=(16, 9))

for i in range(n):
    r = i // cols
    c = i % cols
    ax = axes[r][c]
    plot_spectrogram(example_spectrograms[i].numpy(), ax)
    ax.set_title(label_names[example_spect_labels[i].numpy()])

# plt.show()

train_spectrogram_ds = train_spectrogram_ds.cache().shuffle(10000).prefetch(tf.data.AUTOTUNE)
val_spectrogram_ds = val_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)
test_spectrogram_ds = test_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)

input_shape = example_spectrograms.shape[1:]
print('Input shape:', input_shape)
num_labels = len(label_names)

# Instantiate the `tf.keras.layers.Normalization` layer.
norm_layer = layers.Normalization()
# Fit the state of the layer to the spectrograms
# with `Normalization.adapt`.
norm_layer.adapt(data=train_spectrogram_ds.map(map_func=lambda spec, label: spec))

print("check dataset creation: ")
for example_spectrograms, example_spect_labels in train_spectrogram_ds.take(1):
    print(example_spectrograms.shape, example_spect_labels.shape)

for example_spectrograms, example_spect_labels in val_spectrogram_ds.take(1):
    print(example_spectrograms.shape, example_spect_labels.shape)


model = models.Sequential([
    layers.Input(shape=input_shape),
    # Downsample the input.
    layers.Resizing(32, 32),
    # Normalize.
    norm_layer,
    layers.Conv2D(32, 3, activation='relu'),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_labels),
])

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)



print("train spectrogram dataset: ", train_spectrogram_ds)
print("validate spectrogram dataset: ", val_spectrogram_ds)
print("Number of elements in the training dataset:", tf.data.experimental.cardinality(train_spectrogram_ds).numpy())
print("Number of elements in the validation dataset:", tf.data.experimental.cardinality(val_spectrogram_ds).numpy())

for batch in val_spectrogram_ds.take(1):
    inputs, targets = batch
    print("Validation Batch Shapes:", inputs.shape, targets.shape)

EPOCHS = 10
model.fit(
    train_spectrogram_ds,
    validation_data=val_spectrogram_ds,
    epochs=EPOCHS,

)



model.evaluate(test_spectrogram_ds, return_dict=True)

y_pred = model.predict(test_spectrogram_ds)

y_pred = tf.argmax(y_pred, axis=1)

y_true = tf.concat(list(test_spectrogram_ds.map(lambda s, lab: lab)), axis=0)

confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx,
            xticklabels=label_names,
            yticklabels=label_names,
            annot=True, fmt='g')
plt.xlabel('Prediction')
plt.ylabel('Label')
# plt.show()

# x = pathlib.Path('data/Mitch Dataset/Snare/Snares-01.wav')
x = pathlib.Path('data/Mitch Dataset/Crash/Crashes-11.wav')
x = tf.io.read_file(str(x))
x, sample_rate = tf.audio.decode_wav(x, desired_channels=1, desired_samples=44100, )
x = tf.squeeze(x, axis=-1)
waveform = x
x = get_mel_spectrogram(x)
x = x[tf.newaxis, ...]

prediction = model(x)
x_labels = ['Crash', 'HH_Closed', 'HH_Open', 'Kick', 'Ride', 'Snare',
               'Tom_Lo', 'Tom_Med', 'Tom_Hi']
# x_labels = ['Kick', 'Toms', 'Overheads', 'Snare']
#x_labels = ['China', 'Crash', 'HH_Bell', 'HH_Closed',
#            'HH_Closed + Kick', 'HH_Closed + Snare', 'HH_Open', 'HH_Semi', 'Kick', 'Ride',
#            'Ride_Bell', 'Snare', 'Splash', 'Stack', 'Tom_Hi',
#            'Tom_Lo', 'Tom_Med']
plt.bar(x_labels, tf.nn.softmax(prediction[0]))
plt.title('Crash')
plt.show()

display.display(display.Audio(waveform, rate=44100))

class ExportModel(tf.Module):
    def __init__(self, model):
        self.model = model

        # Accept either a string-filename or a batch of waveforms.
        # You could add additional signatures for a single wave, or a ragged-batch.
        self.__call__.get_concrete_function(
            x=tf.TensorSpec(shape=(), dtype=tf.string))
        self.__call__.get_concrete_function(
            x=tf.TensorSpec(shape=[1, None], dtype=tf.float32))

    @tf.function
    def __call__(self, x):
        # If they pass a string, load the file and decode it.
        if x.dtype == tf.string:
            x = tf.io.read_file(x)
            x, _ = tf.audio.decode_wav(x, desired_channels=1, desired_samples=44100, )
            x = tf.squeeze(x, axis=-1)
            x = x[tf.newaxis, :]

        x = get_mel_spectrogram(x)  # Assuming `get_spectrogram` is the function to compute mel spectrogram
        result = self.model(x, training=False)

        class_ids = tf.argmax(result, axis=-1)
        class_names = tf.gather(label_names, class_ids)
        return {'predictions': result,
                'class_ids': class_ids,
                'class_names': class_names}


data_dir = 'data/Simple Tracks-07.wav'

# Export the model, feed the audio file you want to send to the model
export = ExportModel(model)
export(tf.constant(str(data_dir)))

# Save the model
tf.saved_model.save(export, "saved")

# Load in the audio file, first 17 seconds
samples, sr = librosa.load('data/Simple Tracks-07.wav')

# Detect beats and calculate BPM
tempo, _ = librosa.beat.beat_track(y=samples, sr=sr)
print(f"BPM: {tempo}")

beat_duration = 60 / tempo  # duration of one beat in seconds

# Define the number of beats per measure (adjust based on your preference)
beats_per_measure = 4

# Calculate measure duration
measure_duration = beat_duration * beats_per_measure
print(f"Measure Duration: {measure_duration} seconds")

# Get the timestamps of each hit or each beat
onset_times = librosa.onset.onset_detect(y=samples, sr=sr, units='time')

# Create a list to hold the segmented audio
segments = []

# Define the maximum segment duration in seconds
max_segment_duration = 0.3

# Calculate the maximum number of samples for the segment duration
max_segment_samples = int(max_segment_duration * sr)

# model = tf.keras.models.load_model("drum_model")

label_names = ['Crash', 'HH_Closed', 'HH_Open', 'Kick', 'Ride', 'Snare',
               'Tom_Lo', 'Tom_Med', 'Tom_Hi']

for i, onset_time in enumerate(onset_times):
    # Calculate the start sample for the segment
    start_sample = int(onset_time * sr)

    # Calculate the end sample for the segment (no more than 0.5 seconds after the start)
    end_sample = min(start_sample + max_segment_samples, len(samples))

    # Extract the segment from the audio
    segment = samples[start_sample:end_sample]

    # Make a prediction for the segment
    x = librosa.feature.melspectrogram(y=segment, sr=sr)
    x = np.expand_dims(x, axis=0)
    prediction = model.predict(x)

    # Process the prediction
    threshold = 0.5  # Adjust the threshold as needed, lower number = more sensitive
    predicted_probs = tf.nn.softmax(prediction[0])
    predicted_labels = [label_names[i] for i, prob in enumerate(predicted_probs) if prob > threshold]

    # Add the segment to the list
    segments.append(segment)

    # Check if any notes were predicted above the threshold
    if predicted_labels:
        print(f"Segment {i + 1}: Predicted Labels - {predicted_labels}")
    else:
        print(f"Segment {i + 1}: No notes predicted above the threshold")


'''
Export model test
'''

# Load the exported model
exported_model = tf.saved_model.load("saved")


# Load in the audio file, first 17 seconds
samples, sr = librosa.load('data/Simple Tracks-07.wav')

# Detect beats and calculate BPM
tempo, _ = librosa.beat.beat_track(y=samples, sr=sr)
print(f"BPM: {tempo}")

beat_duration = 60 / tempo  # duration of one beat in seconds

# Define the number of beats per measure (adjust based on your preference)
beats_per_measure = 4

# Calculate measure duration
measure_duration = beat_duration * beats_per_measure
print(f"Measure Duration: {measure_duration} seconds")

# Get the timestamps of each hit or each beat
onset_times = librosa.onset.onset_detect(y=samples, sr=sr, units='time')

# Create a list to hold the segmented audio
segments = []

# Define the maximum segment duration in seconds
max_segment_duration = 0.5

# Calculate the maximum number of samples for the segment duration
max_segment_samples = int(max_segment_duration * sr)

label_names = ['Crash', 'HH_Closed', 'HH_Open', 'Kick', 'Ride', 'Snare',
               'Tom_Lo', 'Tom_Med', 'Tom_Hi']

for i, onset_time in enumerate(onset_times):
    # Calculate the start sample for the segment
    start_sample = int(onset_time * sr)

    # Calculate the end sample for the segment (no more than 0.5 seconds after the start)
    end_sample = min(start_sample + max_segment_samples, len(samples))

    # Extract the segment from the audio
    segment = samples[start_sample:end_sample]

    # Make a prediction for the segment using the exported model
    prediction = exported_model(tf.constant(segment[tf.newaxis, :]))

    # Process the prediction
    threshold = 0.5  # Adjust the threshold as needed, lower number = more sensitive
    predicted_labels = prediction['class_names'].numpy()

    # Add the segment to the list
    segments.append(segment)

    # Check if any notes were predicted above the threshold
    if predicted_labels:
        print(f"Segment {i + 1}: Predicted Labels - {predicted_labels}")
    else:
        print(f"Segment {i + 1}: No notes predicted above the threshold")

'''
# Load your full-length drum track
drum_track_path = 'data/Test Track 1-2.wav'
drum_track, sr = librosa.load(drum_track_path, sr=None)

# Detect onsets
onset_frames = librosa.onset.onset_detect(y=drum_track, sr=sr)

# Convert onset frames to times
onset_times = librosa.frames_to_time(onset_frames, sr=sr)

# Define the duration of each segment (adjust as needed)
segment_duration = 0.5  # in seconds
# Maybe later implement code to automatically calculate segment duration
# eg find the BPM, 120 BPM = 0.5 seconds per quarter note

# Convert segment duration to samples
segment_length = int(segment_duration * sr)

# Segment the audio based on onsets
segments = [drum_track[start:start + segment_length] for start in onset_frames]

# Print the segments
for i, segment in enumerate(segments):
    print(f"Segment {i + 1}: Duration - {len(segment) / sr} seconds")
'''

