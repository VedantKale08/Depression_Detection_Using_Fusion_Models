import torch
import torch.nn as nn

class DepressionHybridModel(nn.Module):
    def __init__(self, input_size=210, hidden_size=64, num_layers=1, dropout=0.5):
        """
        Multimodal Sequence Model for Depression Detection based on Audio, Text, and Facial features.
        
        Args:
            input_size (int): Temporal features per timestep — COVAREP(74)+CLNF(136)=210.
            hidden_size (int): Number of nodes inside the hidden state of the LSTM.
            num_layers (int): Depth of the LSTM.
            dropout (float): Dropout probability between LSTM layers.
        """
        super(DepressionHybridModel, self).__init__()
        
        # We use batch_first=True so input expects (Batch, Seq_Len, Features)
        # Input: COVAREP (74) + CLNF (136) = 210 temporal features per frame
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
        x:           (Batch, Time_Steps, 210)  — temporal COVAREP + CLNF features
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

if __name__ == "__main__":
    print("Testing Model Dimensionality...")
    model = DepressionHybridModel(input_size=210, hidden_size=64, num_layers=1, dropout=0.5)
    dummy_x = torch.randn(8, 300, 210)   # Batch=8, 300 frames, 210 temporal features
    dummy_emo = torch.randn(8, 14)        # Batch=8, 14-dim emotion vector
    dummy_output = model(dummy_x, dummy_emo)
    print(f"Input X Shape:       {dummy_x.shape}")
    print(f"Input Emotion Shape: {dummy_emo.shape}")
    print(f"Output Shape:        {dummy_output.shape}")
    print("Test Successful!")
