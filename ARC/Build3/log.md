# Architecture Overview:

Train Examples
     ↓
Encode (Vision Model)
     ↓
Learn Transformation
     ↓
Adapt on Test (Test-time training)
     ↓
Predict Output Grid

# Key Components:

1. Grid → Image representation
2. Neural model (CNN / Transformer)
3. Task-specific training (very important)
4. Prediction