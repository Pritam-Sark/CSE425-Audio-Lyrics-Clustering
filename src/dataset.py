import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import librosa
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re

class AudioDataset(Dataset):
    def __init__(self, csv_path, audio_dir, target_sample_rate=22050, duration=30, n_mfcc=13):
        self.metadata = pd.read_csv(csv_path)
        self.audio_dir = audio_dir
        
        # Format ID
        self.metadata['track_id_str'] = self.metadata['track_id'].apply(lambda x: str(x).zfill(6))
        
        # Filter: Keep only files that exist
        def check_path(row):
            path = os.path.join(self.audio_dir, str(row['genre']), f"{row['track_id_str']}.mp3")
            return path if os.path.exists(path) else None

        self.metadata['file_path'] = self.metadata.apply(check_path, axis=1)
        self.metadata = self.metadata.dropna(subset=['file_path']).reset_index(drop=True)
        
        # Audio Settings
        self.target_sample_rate = target_sample_rate
        self.duration = duration
        self.n_mfcc = n_mfcc

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        file_path = self.metadata.loc[idx, 'file_path']
        try:
            # Load and Pad Audio
            audio, sr = librosa.load(file_path, sr=self.target_sample_rate, duration=self.duration)
            target_len = self.target_sample_rate * self.duration
            if len(audio) < target_len:
                audio = np.pad(audio, (0, target_len - len(audio)))
            else:
                audio = audio[:target_len]
                
            # MFCC
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc)
            mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-6) # Standardize
            return torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0), self.metadata.loc[idx, 'track_id']
        except:
            return torch.zeros(1), -1

class LyricsDataset(Dataset):
    def __init__(self, csv_path, lyrics_dir, max_features=1000):
        self.metadata = pd.read_csv(csv_path)
        self.lyrics_dir = lyrics_dir
        self.metadata['track_id_str'] = self.metadata['track_id'].apply(lambda x: str(x).zfill(6))

        # Filter: Keep only files that exist
        def check_path(row):
            path = os.path.join(self.lyrics_dir, str(row['genre']), f"{row['track_id_str']}.txt")
            return path if os.path.exists(path) else None

        self.metadata['file_path'] = self.metadata.apply(check_path, axis=1)
        self.metadata = self.metadata.dropna(subset=['file_path']).reset_index(drop=True)

        # Vectorize Text (TF-IDF)
        print("Vectorizing Lyrics (this usually takes 10-20 seconds)...")
        corpus = []
        for p in self.metadata['file_path']:
            try:
                with open(p, 'r', encoding='utf-8') as f:
                    text = re.sub(r'[^a-zA-Z\s]', '', f.read().lower()) # Clean
                    corpus.append(text)
            except:
                corpus.append("")
        
        self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
        self.features = self.vectorizer.fit_transform(corpus).toarray()
        print(f"Lyrics Vectorized. Shape: {self.features.shape}")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # Return Vector and ID (to match with audio later)
        return torch.tensor(self.features[idx], dtype=torch.float32), self.metadata.loc[idx, 'track_id']