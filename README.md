# Capstone 11 — AI-Based Renewable Energy Source Selection Assistant

## What is included
- `data/simulated_renewable_energy.csv`: 1,800-row simulated educational dataset.
- `train_model.py`: preprocessing, train/test split, Decision Tree vs Random Forest vs baseline comparison, weighted F1/accuracy, confusion matrix.
- `app.py`: Streamlit dashboard for input-based prediction and graphs.
- `outputs/`: generated when training runs.
- `Project_Report.docx`: project report.
- `Capstone_Presentation.pptx`: presentation slides.
- `Viva_Questions.txt`: quick viva preparation.

## Important dataset note
The dataset is **simulated**, not real measured weather or energy-generation data. The labels are generated from simplified illustrative formulas to demonstrate the machine-learning workflow. Do not present its scores as real-world performance.

## Run on a laptop / Google Colab
1. Install Python 3.10+.
2. Open a terminal in this folder and run:
   `pip install -r requirements.txt`
3. Train and compare models:
   `python train_model.py`
4. Start dashboard:
   `streamlit run app.py`
5. Open the local URL printed by Streamlit.

## Google Colab alternative
Upload the CSV, install packages using `!pip install -r requirements.txt`, and run the code in `train_model.py` after adjusting the dataset path. For a live Streamlit app from Colab, a separate tunnelling setup is needed.

## Model workflow
Inputs: solar irradiance, wind speed, temperature, humidity, cloud cover, hour, month.
Target classes: Solar, Wind, Both, Low Potential.
Metrics: accuracy and weighted F1 score on held-out test data.
Model selection: choose the model with the highest weighted F1 among Decision Tree and Random Forest.
