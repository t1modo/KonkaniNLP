# Konkani Idiom and Metaphor Classification

This repository contains code for classifying idiomatic expressions and metaphors in the Konkani language using a **BERT-LSTM** neural network architecture.

---

## 📊 Dataset

The project uses the **Konidioms Corpus** stored in `dataset/Konkani_Dataset.xlsx`. This is the first dataset of idioms in the Konkani language and consists of:

- **6,520 total sentences**
  - 4,399 idiomatically sensed sentences  
  - 2,121 literally sensed sentences
- **A subset of 500 sentences annotated for metaphors**
  - 117 metaphorical  
  - 383 non-metaphorical

Each entry in the dataset includes:
- The idiomatic expression in Konkani
- The sentence using the expression
- Labels indicating whether the usage is literal or idiomatic
- Labels indicating whether the expression is metaphorical
- Split designation (train/test)

---

## 🧱 Code Structure

### 1. IMPORTS & SETUP
- Imports libraries:  
  - PyTorch  
  - Transformers (HuggingFace)  
  - Scikit-learn  
- Sets configuration parameters:  
  - Maximum sequence length  
  - Batch size  

---

### 2. LOAD & PREPROCESS DATA
- Loads the dataset from Excel
- Displays basic statistics
- Prepares data for:
  - Idiom detection
  - Metaphor detection
- Creates balanced subset for metaphor task (100 metaphors / 100 non-metaphors)
- Splits into training and testing sets

---

### 3. TOKENIZER + DATASET
- Initializes **Multilingual BERT (mBERT)** tokenizer
- Defines custom dataset class
- Creates PyTorch `DataLoader` objects
- Handles:
  - Text encoding  
  - Label conversion  

---

### 4. MODEL: mBERT + BiLSTM
- Defines hybrid model:  
  - mBERT for contextual embeddings  
  - BiLSTM for sequential pattern learning
- Creates separate models for:
  - Idiom classification
  - Metaphor classification
- Initializes optimizers and loss functions
- Supports GPU acceleration

---

### 5. TRAINING & VALIDATION
- Defines functions for:
  - Training one epoch
  - Validation
  - Evaluation with classification metrics

---

### 6. EARLY STOPPING TRAINING
- Implements early stopping to prevent overfitting
- Trains both idiom and metaphor models
- Monitors validation loss
- Saves best performing models
- Final performance evaluation

---

### 7. ATTENTION HEAD IMPORTANCE ANALYSIS
- Analyzes BERT attention head importance
- Tracks gradients during backpropagation
- Measures head contributions
- Visualizes results with heatmaps

---

### 8. ANALYZE MODELS SEPARATELY
- Applies attention head analysis to:
  - Idiom model
  - Metaphor model
- Generates separate visualizations

---

### 9. ATTENTION HEAD PRUNING & EVALUATION
- Prunes less important attention heads
- Evaluates pruned vs. original models
- Summarizes:
  - Pruning results
  - Impact on model performance

---

## ✅ Results

- **Idiom identification accuracy**: 82%  
- **Metaphor identification accuracy**: 88%  
- **Attention head pruning**:  
  - ~8.33% of attention heads could be removed with minimal impact on performance

---