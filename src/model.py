import torch
import torch.nn as nn

class DepressionHybridModel(nn.Module):
    def __init__(self, input_size=224, hidden_size=64, num_layers=1, dropout=0.5):
        """
        Multimodal Sequence Model for Depression Detection based on Audio, Text, and Facial features.
        
        Args:
            input_size (int): Expected features per timestep (default 224).
            hidden_size (int): Number of nodes inside the hidden state of the LSTM.
            num_layers (int): Depth of the LSTM.
            dropout (float): Dropout probability between LSTM layers.
        """
        super(DepressionHybridModel, self).__init__()
        
        # We use batch_first=True so input expects (Batch, Seq_Len, Features)
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
        
        # Add a dense network head for the final classification
        self.fc1 = nn.Linear(hidden_size * 2, 64)
        self.relu = nn.ReLU()
        self.dropout_fc = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 1)
        
    def forward(self, x):
        """
        Forward pass.
        x shape: (Batch_Size, Time_Steps, Features)
        """
        # out shape: (batch_size, seq_length, hidden_size * 2)
        out, _ = self.lstm(x)
        
        # Apply Temporal Attention to find the most emotionally relevant frames
        # attention_scores shape: (batch_size, seq_length, 1)
        attention_scores = self.attention_weights(out)
        attention_weights = torch.softmax(attention_scores, dim=1) # softmax over sequence length
        
        # Multiply weights by outputs: (batch_size, seq_length, hidden_size * 2)
        # Sum over sequence length: (batch_size, hidden_size * 2)
        context_vector = torch.sum(attention_weights * out, dim=1)
        
        # Normalize and pass through dense layers
        norm_out = self.layer_norm(context_vector)
        
        dense_out = self.fc1(norm_out)
        dense_out = self.relu(dense_out)
        dense_out = self.dropout_fc(dense_out)
        
        # Final Logits (Binary Cross Entropy with Logits Loss will handle the Sigmoid)
        logits = self.fc2(dense_out)
        
        # Remove empty dimensions so output matches label shape (Batch,)
        return logits.squeeze(1)

if __name__ == "__main__":
    # Dry run functionality check
    print("Testing Model Dimensionality...")
    model = DepressionHybridModel()
    dummy_input = torch.randn(8, 300, 224) # Batch of 8, 300 frames, 224 features
    dummy_output = model(dummy_input)
    print(f"Input Shape: {dummy_input.shape}")
    print(f"Output Shape: {dummy_output.shape}") 
    print("Test Successful!")
