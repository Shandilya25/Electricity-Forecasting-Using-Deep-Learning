
# 🔁 Dual RNN Project

This project implements a Dual RNN (Recurrent Neural Network) for sequence modeling tasks using PyTorch.  
The notebook contains code for data preprocessing, training the model, saving it, and reloading it for inference.
#### Architecture
(Architecture.png)

---

## ✅ Step 1: Install Dependencies

Make sure  the code is executed  in **Jupyter Notebook**.

Use the following commands inside a Jupyter Notebook cell to install required libraries:

```python
!pip install torch
!pip install pandas
!pip install numpy
!pip install matplotlib
!pip install scikit-learn
```

These install all necessary packages for training, data handling, and visualization.

---

---

## ▶️ Step 1: Run the Code Step-by-Step

1. Run all notebook cells from **top to bottom**.
2. The notebook performs the following steps:
   - Imports required libraries
   - Loads and preprocesses data
   - Initializes the Dual RNN model
   - Trains the model
   - Saves trained model to the `models/` folder
   - Optionally, loads saved model and runs inference or evaluation

---

## 💾 Step 2: Load a Saved Model (if training is already done)

If existing saved model is to be loaded,it can be loaded it using:(The saved models are in the directory 'Models')

```python
import torch

# Initialize the model (use same parameters as training)
model = DualRNN(input_dim=..., hidden_dim=..., output_dim=..., num_layers=...)

# Load the trained parameters
model.load_state_dict(torch.load("models/dual_rnn_model.pth"))
model.eval()  # Set the model to evaluation mode
```

> Replace `...` with the correct values you used during training.

---

## 🔄 Step 3: Re-training or Modifying the Model

- If we want to retrain from scratch, execute the training cells again.
- To tweak the architecture, modify `DualRNN` class inside `model.py`.

---



---

## 📂 Project Folder Structure

```
project_folder/
├── Code.ipynb             # Main Jupyter notebook
├── Models/                # Folder for saved model weights
└── README.md              # This readme file
└── Graphs                 #Contains graph plots
```

---

