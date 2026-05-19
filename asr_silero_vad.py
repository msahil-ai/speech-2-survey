import torch
import torchaudio
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

def process_and_save_speech(input_path, output_path):
    print("Loading VAD model...")
    # Load the Silero VAD model
    model = load_silero_vad()
    
    # Read the audio file (read_audio outputs a 1D tensor and forces 16kHz)
    wav = read_audio(input_path)
    
    print("Analyzing audio for speech...")
    # Detect speech timestamps
    # We leave return_seconds as False (default) to get exact sample indices for clean cutting
    speech_timestamps = get_speech_timestamps(
        wav,
        model
    )
    
    print(f"Detected {len(speech_timestamps)} speech segments.")

    # Fallback: If no speech is found, just save the original to prevent pipeline crashes
    if not speech_timestamps:
        print("Warning: No speech detected! Saving original file as fallback.")
        torchaudio.save(output_path, wav.unsqueeze(0), 16000)
        return

    # Slice out the dead air
    speech_chunks = []
    for segment in speech_timestamps:
        # Extract just the speech portion using the exact sample indices
        chunk = wav[segment['start'] : segment['end']]
        speech_chunks.append(chunk)
        
    # Stitch all the clean speech chunks together into a single continuous track
    gapless_audio = torch.cat(speech_chunks)
    
    # Save the new gapless audio file
    # torchaudio.save expects a 2D tensor [channels, samples], so we use unsqueeze(0) to add the channel back
    torchaudio.save(output_path, gapless_audio.unsqueeze(0), 16000)
    print(f"Success! Gapless audio saved to: {output_path}")

if __name__ == "__main__":
    
    audio_file = r'enhanced_audio.wav'
    output_file = r'vad_processed_audio.wav'
    
    process_and_save_speech(audio_file, output_file)