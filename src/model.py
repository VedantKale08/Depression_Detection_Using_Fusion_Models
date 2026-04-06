import torch
import torch.nn as nn

class DepressionHybridModel(nn.Module):
    def __init__(self, input_size=112, hidden_size=64, num_layers=1, dropout=0.5):
        """
        Multimodal Sequence Model for Depression Detection based on Audio, Text, and Facial features.
        
        Args:
            input_size (int): Temporal features per timestep — COVAREP(74)+AUs(20)+Pose(6)+Gaze(12)=112.
            hidden_size (int): Number of nodes inside the hidden state of the LSTM.
            num_layers (int): Depth of the LSTM.
            dropout (float): Dropout probability between LSTM layers.
        """
        super(DepressionHybridModel, self).__init__()
        
        # We use batch_first=True so input expects (Batch, Seq_Len, Features)
        # Input: COVAREP (74) + AUs (20) + Pose (6) + Gaze (12) = 112 temporal features per frame
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Temporal Attention Mechanism
        self.attention_weights = nn.Linear(hidden_size * 2, 1)
        
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        
        # Emotion auxiliary dimension: audio(7) + text(7) = 14
        # Concatenated with LSTM context vector AFTER temporal processing
        emotion_dim = 14
        
        # Dense head: LSTM context (hidden*2) + emotion aux (14) -> 64 -> 1
        self.fc1 = nn.Linear(hidden_size * 2 + emotion_dim, 64)
        self.relu = nn.ReLU()
        self.dropout_fc = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 1)
        
    def forward(self, x, emotion_vec):
        """
        Forward pass.
        x:           (Batch, Time_Steps, 112)  — temporal features
        emotion_vec: (Batch, 14)               — session-level audio(7) + text(7) emotions
        """
        # LSTM over temporal sequence
        # out shape: (batch_size, seq_length, hidden_size * 2)
        out, _ = self.lstm(x)
        
        # Temporal Attention: find the most emotionally relevant frames
        attention_scores = self.attention_weights(out)      # (batch, seq_len, 1)
        attention_weights = torch.softmax(attention_scores, dim=1)
        context_vector = torch.sum(attention_weights * out, dim=1)  # (batch, hidden*2)
        
        # LayerNorm on temporal context
        norm_out = self.layer_norm(context_vector)          # (batch, hidden*2)
        
        # Fuse: concatenate LSTM context with session-level emotion features
        # This lets the model use emotion as auxiliary signal WITHOUT leaking participant identity
        fused = torch.cat([norm_out, emotion_vec], dim=1)  # (batch, hidden*2 + 14)
        
        dense_out = self.fc1(fused)
        dense_out = self.relu(dense_out)
        dense_out = self.dropout_fc(dense_out)
        
        # Final logit (BCEWithLogitsLoss handles sigmoid)
        logits = self.fc2(dense_out)
        return logits.squeeze(1)

import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :].unsqueeze(0)
        return x

class DepressionTransformerModel(nn.Module):
    def __init__(self, input_size=112, d_model=128, nhead=4, num_layers=2, dropout=0.5):
        """
        Transformer-based Multimodal Model for Depression Detection.
        """
        super(DepressionTransformerModel, self).__init__()
        
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model*2, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.layer_norm = nn.LayerNorm(d_model)
        
        emotion_dim = 14
        self.fc1 = nn.Linear(d_model + emotion_dim, 64)
        self.relu = nn.ReLU()
        self.dropout_fc = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 1)
        
    def forward(self, x, emotion_vec):
        # x: (Batch, Time_Steps, 112)
        x = self.input_projection(x) # (Batch, Time_Steps, d_model)
        x = self.pos_encoder(x)
        
        # Transformer Context
        x = self.transformer_encoder(x) # (Batch, Time_Steps, d_model)
        
        # Average pooling over time steps
        context_vector = torch.mean(x, dim=1) # (Batch, d_model)
        norm_out = self.layer_norm(context_vector)
        
        # Fuse with emotion
        fused = torch.cat([norm_out, emotion_vec], dim=1) # (Batch, d_model + 14)
        
        dense_out = self.fc1(fused)
        dense_out = self.relu(dense_out)
        dense_out = self.dropout_fc(dense_out)
        
        logits = self.fc2(dense_out)
        return logits.squeeze(1)

if __name__ == "__main__":
    print("Testing Model Dimensionality...")
    dummy_x = torch.randn(8, 300, 112)   # Batch=8, 300 frames, 112 temporal features
    dummy_emo = torch.randn(8, 14)        # Batch=8, 14-dim emotion vector
    
    print("\n[Bi-LSTM Model]")
    model_lstm = DepressionHybridModel(input_size=112, hidden_size=64, num_layers=1, dropout=0.5)
    out_lstm = model_lstm(dummy_x, dummy_emo)
    print(f"Output Shape:        {out_lstm.shape}")
    
    print("\n[Transformer Model]")
    model_tx = DepressionTransformerModel()
    out_tx = model_tx(dummy_x, dummy_emo)
    print(f"Output Shape:        {out_tx.shape}")
    print("Test Successful!")
