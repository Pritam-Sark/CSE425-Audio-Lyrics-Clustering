import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import librosa
import numpy as np

class AudioDataset(Dataset):
    def __init__(self, csv_path, audio_dir, target_sample_rate=22050, duration=30, n_mfcc=13):
        """
        Args:
            csv_path (str): Path to the csv file.
            audio_dir (str): Root path to audio folders (e.g. "D:/CSE425 Project/data/audio")
        """
        self.metadata = pd.read_csv(csv_path)
        self.audio_dir = audio_dir
        self.target_sample_rate = target_sample_rate
        self.duration = duration
        self.n_mfcc = n_mfcc
        
        # 1. Format track_id to have leading zeros (e.g., 5 -> "000005")
        self.metadata['track_id_str'] = self.metadata['track_id'].apply(lambda x: str(x).zfill(6))
        
        # 2. Check which files exist (Handling Genre Subfolders)
        # Structure: audio_dir / Genre / 000005.mp3
        def check_path(row):
            path = os.path.join(self.audio_dir, str(row['genre']), f"{row['track_id_str']}.mp3")
            if os.path.exists(path):
                return path
            return None

        self.metadata['file_path'] = self.metadata.apply(check_path, axis=1)
        
        # 3. Filter out missing files
        initial_count = len(self.metadata)
        self.metadata = self.metadata.dropna(subset=['file_path']).reset_index(drop=True)
        print(f"Dataset initialized. Found {len(self.metadata)} valid audio files out of {initial_count}.")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        file_path = self.metadata.loc[idx, 'file_path']
        
        try:
            # Load Audio (limit to 30 seconds)
            audio, sr = librosa.load(file_path, sr=self.target_sample_rate, duration=self.duration)
            
            # Ensure fixed length (padding or truncation)
            target_length = self.target_sample_rate * self.duration
            if len(audio) < target_length:
                audio = np.pad(audio, (0, target_length - len(audio)))
            else:
                audio = audio[:target_length]

            # Extract MFCCs
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc)
            
            # Standardize (Mean=0, Std=1) - Crucial for VAEs
            mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-6)
            
            # Flatten to 1D vector
            return torch.tensor(mfcc.flatten(), dtype=torch.float32)
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return torch.zeros(1) # Safety fallback