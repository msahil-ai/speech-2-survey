import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
from df.enhance import enhance, init_df, load_audio, save_audio

def denoise_audio(input_path, output_path):
    # init_df returns the model, state, and the expected sample rate
    model, df_state, _ = init_df()  
    
    # We use '_' to throw away the metadata object returned by load_audio
    audio, _ = load_audio(input_path, sr=df_state.sr())
    
    # Denoise the audio
    enhanced_audio = enhance(model, df_state, audio)
    
    # Save the enhanced audio using the integer sample rate from the model state
    save_audio(output_path, enhanced_audio, df_state.sr())
    
    print(f"Successfully denoised and saved to {output_path}")

if __name__ == "__main__":
    
    noisy_file = r'audio_sample\sample_audio.wav'
    clean_file = 'enhanced_audio.wav'

    denoise_audio(noisy_file, clean_file)