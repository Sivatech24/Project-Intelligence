import torch
import torch.nn as nn
import torch.optim as optim

class RecursiveTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=4, vocab_size=1000):
        super(RecursiveTransformer, self).__init__()
        self.d_model = d_model
        
        # 1. The core Transformer Block (4x layers as per diagram)
        # We use batch_first=True for easier dimension handling (Batch, Seq, Feature)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_block = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 2. Input Embedding (for the Question x)
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 3. Learnable initial states for Prediction (y) and Latent Reasoning (z)
        self.initial_y = nn.Parameter(torch.randn(1, 1, d_model))
        self.initial_z = nn.Parameter(torch.randn(1, 1, d_model))
        
        # 4. State Update Heads (MLPs to update the states based on transformer output)
        self.update_z_head = nn.Linear(d_model, d_model)
        self.update_y_head = nn.Linear(d_model, d_model)
        
        # 5. Reverse Embedding (maps final prediction back to vocabulary space for Cross-Entropy Loss)
        self.reverse_embedding = nn.Linear(d_model, vocab_size)

    def forward(self, x, N_sup=16, n_inner=2):
        """
        x: Input tensor of shape (Batch_Size, Sequence_Length)
        N_sup: Number of outer loops (improving prediction y)
        n_inner: Number of inner loops (improving latent z)
        """
        B = x.size(0) # Batch size
        
        # Embed the input question
        x_emb = self.embedding(x) # Shape: (B, Seq_Len, d_model)
        
        # Expand the initial learnable states to match the batch size
        y = self.initial_y.expand(B, -1, -1) # Shape: (B, 1, d_model)
        z = self.initial_z.expand(B, -1, -1) # Shape: (B, 1, d_model)
        
        # Outer Loop: Applied N_sup times (trying to improve the prediction y)
        for _ in range(N_sup):
            
            # Inner Loop: Step 1, 2, ..., n (Improve the latent z)
            for _ in range(n_inner):
                # Combine inputs: ⊕ (We concatenate them as a sequence for the transformer)
                # Sequence order: [Question tokens, Answer token, Reasoning token]
                combined_input = torch.cat([x_emb, y, z], dim=1) 
                
                # Pass through the 4-layer transformer block
                out = self.transformer_block(combined_input)
                
                # Extract the output corresponding to the 'z' token (the very last token)
                z_out = out[:, -1:, :]
                # Update z (residual connection style)
                z = z + self.update_z_head(z_out)
                
            # Outer Loop Step: Update y given y, z (and x)
            # Re-concatenate with the newly updated z
            combined_input = torch.cat([x_emb, y, z], dim=1)
            out = self.transformer_block(combined_input)
            
            # Extract the output corresponding to the 'y' token (second to last token)
            y_out = out[:, -2:-1, :]
            # Update y
            y = y + self.update_y_head(y_out)

        # Reverse Embedding: Map the final prediction token to logits
        # Squeeze removes the sequence dimension of length 1
        logits = self.reverse_embedding(y.squeeze(1)) 
        
        return logits, y, z


# ==========================================
# Training and Testing with CUDA Support
# ==========================================
if __name__ == "__main__":
    # Check for CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model and move to device
    vocab_size = 1000
    model = RecursiveTransformer(d_model=512, vocab_size=vocab_size).to(device)
    
    # Define optimizer and loss function (Cross-entropy loss as per diagram)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Create dummy data for demonstration
    batch_size = 8
    seq_length = 10
    
    # Random input token IDs and target labels
    dummy_x = torch.randint(0, vocab_size, (batch_size, seq_length)).to(device)
    dummy_labels = torch.randint(0, vocab_size, (batch_size,)).to(device)
    
    print("\n--- Starting Training Step ---")
    model.train()
    optimizer.zero_grad()
    
    # Forward Pass
    # We set N_sup=2 and n_inner=2 here just to make the dummy run faster. 
    # Change N_sup=16 to match the exact diagram specifications.
    logits, final_y, final_z = model(dummy_x, N_sup=2, n_inner=2) 
    
    # Calculate Loss (Cross-entropy loss block from the diagram)
    loss = criterion(logits, dummy_labels)
    
    # Backward Pass and Optimize
    loss.backward()
    optimizer.step()
    
    print(f"Training iteration completed. Loss: {loss.item():.4f}")
    
    print("\n--- Starting Testing/Inference Step ---")
    model.eval()
    with torch.no_grad():
        test_logits, _, _ = model(dummy_x, N_sup=16, n_inner=2)
        predictions = torch.argmax(test_logits, dim=-1)
        print("Testing inference completed successfully.")
        print(f"Sample predictions shape: {predictions.shape}")
