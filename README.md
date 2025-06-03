# Project Setup and Usage Instructions

Follow these steps to clone the repository, set up the Python environment, and work with the project.

## 1. Clone the Repository
```
git clone https://github.com/mmedlin1997/ml-regression.git
cd ml-regression
```
## 2. Activate environment
```python -m venv venv
venv\Scripts\activate   # On Windows
# source venv/bin/activate   # On macOS/Linux
```
## 3. Install dependencies
```pip install --upgrade pip
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
pip install xgboost lightgbm catboost

pip freeze > requirements.txt
```
## 4. Start Jupyter
```
jupyter notebook
```
## 5. Run or Edit code

## 6. Commit and Push changes
```
git add .
git commit -m "Restructured repo with file and folders"
git push origin master
```
## 7. Deactivate environment
```
deactivate
```
# Notebooks

## 1. regression-dataset-comparison.ipynb
This notebook compares the performance of various regression models across different datasets. 
It provides visualizations and metrics to help evaluate which models are most suitable for particular dataset characteristics.
