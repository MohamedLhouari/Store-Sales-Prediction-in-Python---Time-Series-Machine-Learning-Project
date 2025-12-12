# Store Sales Prediction (Python / Time Series ML)

Forecast daily unit sales for thousands of product families across Favorita stores using classical preprocessing and deep learning (Temporal Convolutional Network) in PyTorch.

## Dataset
- Source: Kaggle "Store Sales - Time Series Forecasting" competition (Corporación Favorita).
- Files used (see `Data/`): `train.csv`, `test.csv`, `stores.csv`, `transactions.csv`, `oil.csv`, `holidays_events.csv`.
- Target: `sales` (daily unit sales per store-family combination).

## Project Structure
- `Data/` &rarr; Raw CSV inputs (unzipped from Kaggle download).
- `Untitled-1.ipynb` &rarr; Exploratory analysis, preprocessing, TCN model training, and submission generation.
- `submission.csv` (created after running the notebook) &rarr; Kaggle-ready predictions.

## Environment Setup
1. Install Python 3.9+ (Anaconda recommended).
2. Create and activate a virtual environment:
	```powershell
	python -m venv .venv
	.\.venv\Scripts\activate
	```
3. Install dependencies:
	```powershell
	pip install -r requirements.txt  # create this file or install ad hoc
	```
	Minimum packages: `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `torch`, `torchvision` (optional), `jupyter`.

> **Note**: The notebook automatically falls back to CPU if CUDA (GPU acceleration) is unavailable. No NVIDIA GPU is required.

## Running the Notebook
1. Launch Jupyter:
	```powershell
	jupyter notebook
	```
2. Open `Untitled-1.ipynb`.
3. Run the cells in order:
	- **Exploration**: Inspect store-family combinations and visualize sample series.
	- **Preprocessing**: Pivot sales to a multivariate time series tensor and scale features.
	- **Sequence Generation**: Create sliding windows with 120-day history to forecast 16 days ahead.
	- **Model Training**: Train the TCN model on the training split, monitor RMSE.
	- **Evaluation**: Validate on the hold-out split.
	- **Full Training + Submission**: Retrain on all data and export `submission.csv`.

## Modeling Overview
- Architecture: Temporal Convolutional Network with dilated 1D convolutions and residual-style receptive field growth.
- Loss: Root Mean Squared Error (RMSE) over predicted horizons.
- Scaling: Standardization per time step to stabilize training.
- Horizon: 16-day forecasts per store-family pair.

## Results & Next Steps
- Baseline RMSE values are printed during training and evaluation cells.
- Potential improvements:
  1. Incorporate external regressors (oil prices, holidays, transactions) into the modeling pipeline.
  2. Experiment with alternative architectures (LSTM/Transformer) and hyperparameter tuning.
  3. Add backtesting, cross-validation, and automated feature engineering.

## Credits
Based on the Corporación Favorita dataset released under the Kaggle competition terms. Project maintained by Mohamed Lhouari.
