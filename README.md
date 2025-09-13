# Ground Motion Modelling with Bidirectional Liquid Neural Network (BLiqNet)

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/9320a2d9-229c-4860-85a7-446c2ba1acc7" />


![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-brightgreen.svg)
![PyTorch](https://img.shields.io/badge/pytorch-1.10%2B-red.svg)

## 🔹 Overview
**BLiqNet** is a **Bidirectional Liquid Neural Network (LNN)** designed for **ground motion modeling** in earthquake engineering.  
Unlike conventional regression models or black-box ML approaches, BLiqNet:
- Performs **forward prediction** of response spectra from seismic source–path–site parameters.
- Performs **inverse inference** of input parameters from observed ground-motion records.
- Provides enhanced **interpretability** through **ODE-driven latent dynamics** and sensitivity analyses.

The model has been validated on the **NGA-West2 dataset** and external earthquake datasets (Turkey & Ridgecrest), showing improved accuracy and generalization compared to traditional Ground Motion Models (GMMs) and ML baselines (ANN, KRR, RFR, SVR, Ensemble).

---

## 🔹 Key Features
- ✅ **Bidirectional learning**: forward + inverse inference.  
- ✅ **Liquid Neural Layer** with ODE solver integration.  
- ✅ **Adjoint method** for efficient gradient backpropagation.  
- ✅ **Robust evaluation**: residual analysis, cross-validation, external dataset testing.  
- ✅ **Comparative benchmarking** against standard ML models.  
- ✅ **Computational efficiency analysis** (training vs inference trade-offs).  

---

## 🔹 Model Architecture
The BLiqNet architecture consists of:
1. **Input layer**: Seismic parameters  
   - Magnitude (Mw)  
   - Fault type  
   - Hypocentral depth  
   - Distance metrics (Rjb, log(dist))  
   - Site conditions (log(Vs30))  
   - Direction  

2. **Liquid Neural Layer**  
   - Continuous-time dynamics modeled via ODEs.  
   - Solved with `torchdiffeq` (adjoint method).  

3. **Forward Prediction Branch**  
   - Fully connected layers with ReLU activation.  
   - Outputs 5%-damped spectral acceleration values.  

4. **Backward Prediction Branch**  
   - Fully connected layer mapping hidden state → input reconstruction.  

---

## 🔹 Installation

Clone the repository:
```bash
git clone https://github.com/yourusername/BliqNet.git
cd BliqNet
````

Install dependencies:

```bash
pip install -r requirements.txt
```

Dependencies include:

* `torch`, `torchdiffeq`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib`

---

## 🔹 Usage

### 1. Data Preparation

* Place NGA-West2 or your custom dataset in the `data/` directory.
* Update `input_features` and `output_features` in the script if needed.

### 2. Training the Model

```bash
python train_bliqnet.py
```

This will:

* Preprocess inputs/outputs with MinMax scaling.
* Train BLiqNet for forward + inverse tasks.
* Save the best model checkpoint to `models/best_liquid_model.pth`.

### 3. Inference

```bash
python predict_bliqnet.py
```

Example:

```python
sample_input = torch.tensor(X_scaled[:5], dtype=torch.float32).to(device)
forward_pred, backward_pred = model(sample_input)
```

---

## 🔹 Results

* **Forward prediction**: BLiqNet accurately predicts spectral acceleration across spectral periods.
* **Inverse prediction**: Successfully reconstructs magnitude, distance, and site parameters with R² > 0.9 (5-fold CV).
* **Residual analysis**: Demonstrates minimal bias across Mw, Rjb, and Vs30.
* **Computational trade-off**: Higher training cost, but efficient inference (\~0.10 ms/sample).


---

## 🔹 Project Structure

```
├── data/                  # Dataset storage
├── models/                # Saved models and scalers
├── scripts/               # Helper scripts
│   ├── train_bliqnet.py   # Training script
│   ├── predict_bliqnet.py # Inference script
│   └── utils.py           # Preprocessing, plotting
├── docs/                  # Documentation, figures
├── requirements.txt       # Dependencies
└── README.md              # Project documentation
```

---

## 🔹 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## 🔹 Contact

📧 **Pavan Mohan Neelamraju** – [npavanmohan3@gmail.com](mailto:npavanmohan3@gmail.com)
🌐 [Personal Website](https://pavanmohan.netlify.app)

---

```

