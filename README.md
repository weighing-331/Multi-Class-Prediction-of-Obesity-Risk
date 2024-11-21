# Multi-Class-Prediction-of-Obesity-Risk

## Project Description
This project focuses on predicting obesity risk using advanced machine learning techniques. MATLAB is used for Exploratory Data Analysis (EDA) and preprocessing, while Python is employed for machine learning model training and evaluation. The final deployment includes a UI built in Python to integrate the best-performing model, LightGBM.

## Features
- **EDA with MATLAB**: Data visualization, statistical summaries, feature importance, and preprocessing.
- **Machine Learning in Python**: Implementation and evaluation of Logistic Regression, Random Forest, LightGBM, XGBoost, and AdaBoost.
- **Best Model**: LightGBM identified as the best-performing model for obesity risk prediction.
- **UI for Predictions**: A graphical interface implemented in `GUI.py` for real-time predictions.

---

## File Overview
### MATLAB Scripts
1. **`code_EDA.m`**:  
   - Performs exploratory data analysis (EDA) on the dataset, including:
     - Statistical summaries
     - Data distribution visualization
     - Correlation matrix and PCA analysis
     - Feature importance analysis using Random Forest
     - Class-wise and gender-wise analysis of BMI
   - Saves results (e.g., visualizations) to the `plot` directory.

2. **`code_ML.m`**:  
   - Handles data preprocessing and feature engineering:
     - One-hot encoding of categorical variables
     - Feature scaling and removal of multicollinear features
   - Trains and evaluates multiple machine learning models, including:
     - Decision Tree, Random Forest, SVM, Logistic Regression, KNN, and Naive Bayes
   - Outputs evaluation metrics (accuracy, precision, recall, F1-score) and confusion matrices for each model.

### Python Scripts
1. **`main.py`**:  
   - Implements and evaluates various machine learning models using Python:
     - Logistic Regression, Random Forest, LightGBM, XGBoost, AdaBoost
   - Conducts hyperparameter tuning and outputs evaluation metrics.

2. **`GUI.py`**:  
   - A user interface for real-time obesity risk predictions using the trained LightGBM model.

---

## Large Files
Due to GitHub's size limitations, some large files are hosted on Google Drive:
- [Multi-Class Prediction of Obesity Risk.mp4](https://drive.google.com/file/d/1rj2GHZmXrapIm5v5nIvCS45d5qHKWHtY/view?usp=drive_link)  
  *(A video walkthrough of the project.)*
- [Multi-Class Prediction of Obesity Risk.pptx](https://docs.google.com/presentation/d/1v5y5zPNYwYtG-SQu9l4CuZLmQiTCC2bj/edit?usp=drive_link&ouid=112522920743661236400&rtpof=true&sd=true)  
  *(A presentation detailing methodology and results.)*

---

## Getting Started

### Prerequisites
- **MATLAB R2021b or higher** (for `code_EDA.m` and `code_ML.m`)
- **Python 3.8+** (for `main.py` and `GUI.py`)
- **Libraries**: Install Python dependencies with:
  ```bash
  pip install -r requirements.txt
  ```

---

### Steps to Run

1. **MATLAB Scripts**:
   - Run `code_EDA.m` for exploratory data analysis and visualizations.
   - Run `code_ML.m` for data preprocessing and training machine learning models in MATLAB.

2. **Python Scripts**:
   - Use [Colab Notebook](https://colab.research.google.com/drive/1LCGhvLboFNF7lQ-TUKCeRcW7m4eAEqvC#scrollTo=QvS4dYJ2Gmh9) or run `main.py` locally to train and evaluate machine learning models.
   - Launch the UI with `GUI.py` for real-time obesity risk predictions:
     ```bash
     python GUI.py
     ```

---

## Results
1. **Best Model**: LightGBM achieved the highest performance in multi-class classification.
2. **Visualizations**:  
   - Feature importance (Random Forest)
   - BMI distribution
   - PCA scatter plots
   - Confusion matrices

---

## References
- Kaggle Competition: [Playground Series - Season 4 Episode 2](https://www.kaggle.com/competitions/playground-series-s4e2/overview)

---

## Contributing
Contributions are welcome! Please fork the repository, make changes, and submit a pull request.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

如果需要進一步修改或調整，請告訴我！
