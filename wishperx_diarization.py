import os
import gc
import numpy as np
import soundfile as sf
import whisperx
from dotenv import load_dotenv
from whisperx.diarize import DiarizationPipeline

# --- HELPER FUNCTION ---
def format_and_save_log(diarized_segments, output_filename="speaker_log.txt"):
    """
    Parses WhisperX nested JSON output, prints clean logs to the terminal,
    and saves them to a text file for LLM extraction.
    """
    print("\n--- Final Diarized Transcript ---")
    
    with open(output_filename, "w", encoding="utf-8") as file:
        for segment in diarized_segments:
            # Convert float seconds to MM:SS format
            start_mins, start_secs = divmod(segment['start'], 60)
            end_mins, end_secs = divmod(segment['end'], 60)
            
            time_stamp = f"{int(start_mins):02d}:{start_secs:05.2f} - {int(end_mins):02d}:{end_secs:05.2f}"
            
            # Extract speaker (fallback to UNKNOWN if the model missed it)
            speaker = segment.get('speaker', 'UNKNOWN_SPEAKER')
            
            # Extract the actual text
            text = segment.get('text', '').strip()
            
            # Construct the final clean string
            log_line = f"[{time_stamp}] {speaker}: {text}"
            
            # Output to terminal and file
            print(log_line)
            file.write(log_line + "\n")
            
    print(f"\nTranscript successfully saved to {output_filename}")


# --- MAIN PIPELINE ---
if __name__ == "__main__":
    
    # Load Environment Variables
    load_dotenv()
    hf_token = os.getenv("HF_WISPER_TOKEN")

    # Configuration
    device = "cpu" 
    audio_file = "vad_processed_audio.wav" 
    batch_size = 16 
    compute_type = "int8" 

    print("Loading Whisper model...")
    # 1. Transcribe with Whisper
    model = whisperx.load_model("base", device, compute_type=compute_type) 

    print("Loading audio file...")
    # BYPASS FFmpeg: Read the audio manually using the soundfile library
    audio_data, _ = sf.read(audio_file)
    audio = audio_data.astype(np.float32) # WhisperX expects a float32 numpy array

    print("Transcribing audio...")
    result = model.transcribe(audio, batch_size=batch_size, task="translate") # Use "transcribe" if you want to keep the original language

    print("Aligning transcript...")
    # 2. Align whisper output (forces exact timestamps for the words)
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    print("Running speaker diarization (This may take a moment)...")
    # 3. Assign speaker labels using Pyannote
    diarize_model = DiarizationPipeline(token=hf_token, device=device)
    
    # Note: If you know it's always exactly 1 surveyor and 1 citizen, you can uncomment the next line
    # to force exactly 2 speakers and improve accuracy:
    # diarize_segments = diarize_model(audio, min_speakers=2, max_speakers=2)
    diarize_segments = diarize_model(audio)

    print("Merging timestamps...")
    # 4. Merge text and speaker labels
    result = whisperx.assign_word_speakers(diarize_segments, result)

    # 5. Clean output generation
    format_and_save_log(result["segments"], "speaker_log.txt")