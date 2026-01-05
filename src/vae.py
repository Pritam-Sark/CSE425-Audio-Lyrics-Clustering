import torch
import torch.nn as nn

class ConvVAE(nn.Module):
    def __init__(self, input_shape, latent_dim=64):
        """
        Args:
            input_shape: (channels, height, width) -> (1, 13, 1292)
            latent_dim: Size of the bottleneck vector
        """
        super(ConvVAE, self).__init__()
        self.input_shape = input_shape
        c, h, w = input_shape

        # --- ENCODER (Compress 2D Image -> Latent) ---
        self.encoder = nn.Sequential(
            # Layer 1: Conv2d
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # Layer 2
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # Layer 3
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # Layer 4
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        
        # Calculate size after convolutions to flatten correctly
        # Input (13, 1292) -> reduced by factor of 2 four times (2^4 = 16)
        # 13/16 -> 1 (min size)
        # 1292/16 -> 81
        self.flatten_dim = 256 * 1 * 81 
        
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

        # --- DECODER (Expand Latent -> 2D Image) ---
        self.decoder_input = nn.Linear(latent_dim, self.flatten_dim)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=(0,1)), # Adjust output_padding to match shapes
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=(0,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=(1,0)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=(0,0)), 
            # No activation at end (we output raw values)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # x shape: [Batch, 1, 13, 1292]
        
        # Encoder
        h = self.encoder(x)
        h_flat = h.view(h.size(0), -1)
        
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)
        z = self.reparameterize(mu, logvar)
        
        # Decoder
        d_in = self.decoder_input(z)
        d_reshaped = d_in.view(d_in.size(0), 256, 1, 81)
        
        recon = self.decoder(d_reshaped)
        
        # Resize to match input exactly (in case of rounding errors in Conv logic)
        if recon.shape != x.shape:
             recon = torch.nn.functional.interpolate(recon, size=x.shape[2:])

        return recon, mu, logvar