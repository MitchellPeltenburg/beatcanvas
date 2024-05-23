from django.http import HttpResponse, FileResponse
from django.template import loader

from .models import Member
from music21 import stream, note, duration
import tensorflow as tf
import librosa
import tempfile
from django.views.decorators.csrf import csrf_exempt
import subprocess
from music21 import environment



def convert_musicxml_to_audio(musicxml_path, output_path, musescore_path):
    # Command to convert MusicXML to audio (e.g., WAV)
    command = [
        musescore_path,
        musicxml_path,
        '--export-to', output_path
    ]
    subprocess.run(command, check=True)


def transcription(request):
    mymembers = Member.objects.all().values()
    template = loader.get_template('all_members.html')
    context = {
        'mymembers': mymembers,
    }
    return HttpResponse(template.render(context, request))


def details(request, id):
    mymember = Member.objects.get(id=id)
    template = loader.get_template('details.html')
    context = {
        'mymember': mymember,
    }
    return HttpResponse(template.render(context, request))


@csrf_exempt
def index(request):
    context = {}
    if request.method == 'POST' and request.FILES.get('audio_file'):
        audio_file = request.FILES['audio_file']

        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            for chunk in audio_file.chunks():
                temp_audio.write(chunk)

            # Load the saved model
            exported_model = tf.saved_model.load(
                "C:/Users/mitch/OneDrive/School/DC/Year 4/Sem 2/beat_canvas/transcription/saved")

            # Load the audio file using librosa
            samples, sr = librosa.load(temp_audio.name)

            # Detect onsets
            onset_frames = librosa.onset.onset_detect(y=samples, sr=sr, units='frames')
            onset_times = librosa.frames_to_time(onset_frames, sr=sr)

            # Define note durations based on BPM
            tempo, _ = librosa.beat.beat_track(y=samples, sr=sr)
            durations = {
                'quarter': 60 / tempo,
                'eighth': 30 / tempo,
                '16th': 15 / tempo,
            }

            # Define the mapping of labels to pitch values
            label_to_pitch = {
                'Crash': 49,
                'HH_Closed': 42,
                'HH_Open': 46,
                'Kick': 36,
                'Ride': 51,
                'Snare': 38,
                'Tom_Hi': 50,
                'Tom_Med': 47,
                'Tom_Lo': 45
            }

            # Create an output stream
            output_stream = stream.Stream()

            # Process segments for prediction and quantization
            for i, onset_time in enumerate(onset_times):
                start_sample = int(onset_time * sr)
                if i + 1 < len(onset_times):
                    next_onset_time = onset_times[i + 1]
                    end_sample = int(next_onset_time * sr) - int(0.1 * sr)
                else:
                    end_sample = len(samples)

                segment = samples[start_sample:end_sample]
                segment_duration = librosa.samples_to_time(len(segment), sr=sr)

                prediction = exported_model(tf.constant(segment[tf.newaxis, :]))
                predicted_labels = prediction['class_names'].numpy()

                print(f"Segment {i + 1}: Start Time - {onset_time}, End Time - {librosa.samples_to_time(end_sample, sr=sr)}, Predicted Labels - {predicted_labels}")

                if predicted_labels:
                    for label in predicted_labels:
                        label_str = label.decode('utf-8')
                        pitch = label_to_pitch.get(label_str, None)

                        if pitch:
                            note_obj = note.Note(pitch)
                            note_duration_value = durations['quarter']  # Default to quarter note

                            for dur_label, dur_value in durations.items():
                                if abs(segment_duration - dur_value) <= 0.2:
                                    parts = dur_label.split('-')
                                    base_duration = parts[0]
                                    note_obj.duration.type = base_duration
                                    note_duration_value = dur_value

                                    if len(parts) > 1:
                                        base_duration = parts[1]
                                        modifier = parts[1]
                                        if 'dotted' in modifier:
                                            note_obj.duration.dots = 1
                                        elif 'triplet' in modifier:
                                            note_obj.duration.tuplets = [duration.Tuplet(3, 2)]

                                    break

                            output_stream.append(note_obj)
                            print(f"Note added: {label_str} with duration {note_obj.duration.type}")

                            remaining_duration = segment_duration - note_duration_value
                            while remaining_duration >= durations['16th']:
                                for dur_label, dur_value in sorted(durations.items(), key=lambda x: x[1], reverse=True):
                                    if remaining_duration >= dur_value:
                                        parts = dur_label.split('-')
                                        rest_obj = note.Rest()
                                        if len(parts) > 1:
                                            base_duration = parts[1]
                                            modifier = parts[1]
                                            if 'dotted' in modifier:
                                                rest_obj.duration.dots = 1
                                            elif 'triplet' in modifier:
                                                rest_obj.duration.tuplets = [duration.Tuplet(3, 2)]
                                        else:
                                            base_duration = parts[0]

                                        rest_obj.duration.type = base_duration
                                        rest_duration = dur_value

                                        output_stream.append(rest_obj)
                                        print(f"Rest added with duration {rest_obj.duration.type}")
                                        remaining_duration -= rest_duration
                                        break
                        else:
                            print(f"Unknown label: {label_str}")
                else:
                    print(f"Segment {i + 1}: Start Time - {onset_time}, No notes predicted above the threshold")

            print("Write the output stream to xml file")
            # Save the output stream as a MusicXML file
            output_stream.write('musicxml', fp='media/output.xml')

            # Paths
            musicxml_path = 'media/output.xml'
            output_path = 'media/output.wav'
            musescore_path = 'C:/Program Files/MuseScore 3/bin/MuseScore3.exe'

            # Convert MusicXML to WAV
            print("Converting predicted file to WAV")
            convert_musicxml_to_audio(musicxml_path, output_path, musescore_path)

            env = environment.Environment()

            # Replace 'path/to/musescore' with the actual path to the MuseScore executable on your machine
            env['musescoreDirectPNGPath'] = 'C:/Program Files/MuseScore 3/bin/MuseScore3.exe'
            env['musicxmlPath'] = 'C:/Program Files/MuseScore 3/bin/MuseScore3.exe'
            print("Write output stream to png file")
            output_stream.write('musicxml.png', fp='media/output.png')

            print("Prep file for download")
            # Prepare the file for download
            response = FileResponse(open('media/output.xml', 'rb'))
            response['Content-Disposition'] = 'attachment; filename="output.xml"'
            return response

    print("Made it to page render")
    # Render the index page with the form if it's a GET request or no file is uploaded
    template = loader.get_template('index.html')
    return HttpResponse(template.render())
