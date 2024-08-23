import os
import random
import shutil
import wave

root_dir = "D:\Sebas\Thesis\Datasets\maestro-v3.0.0\maestro-v3.0.0"
output_dir = "D:\Sebas\Thesis\Datasets\less_maestro"

target_duration_ms = 20 * 60 * 60 * 1000  #20 hours

def get_wav_duration(file_path):
    with wave.open(file_path, 'r') as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        duration = (frames / float(rate)) * 1000  # Convert to milliseconds
    return duration

wav_files = []
for subdir, _, files in os.walk(root_dir):
    for file in files:
        if file.endswith('.wav'):
            file_path = os.path.join(subdir, file)
            duration = get_wav_duration(file_path)
            wav_files.append((file_path, duration))

random.shuffle(wav_files)

# Select files until the total duration is approximately 20 hours
selected_files = []
total_duration = 0
for file_path, duration in wav_files:
    if total_duration + duration > target_duration_ms:
        break
    selected_files.append(file_path)
    total_duration += duration

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Copy selected files to the output directory
for file_path in selected_files:
    relative_path = os.path.relpath(file_path, root_dir)
    output_path = os.path.join(output_dir, relative_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    shutil.copy2(file_path, output_path)

print(f"Selected {len(selected_files)} files with a total duration of {total_duration / (1000 * 60 * 60):.2f} hours.")