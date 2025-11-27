# Shell Stock
![shell-oil](/shell.jpg)


## Procedures
- Import the libraries
    - pandas
    - numpy
    - scikit-learn
    - seaborn
    - matplotlib
    - yfianance
- Data Acquisition
    - Data aquired from the Yahoo Finance API
- Feature Engineering
- Pre-Training Visualization
![pre-training-visualization](/output1.png)
- Feature Engineering
- Data Splitting
    - Split the data inot training and testing sets (80% train, 20% tests)
    - Shuffle=False to mainntain the time-series order, avoiding data leakage
- Data Scaling
    - Initialize the StandardScaler
    - Fit the scaler ONLY on the training data to prevent data leakage from the test set
- Model Comparison
    - Logistic Regression
    - K-Nearest Neighbors
    - Support Vector Machine (Linear)
    - Random Forest 
- Model Training 
    - classification report
- Model Evaluation
- Hyperparameter Tuning
- Post-Training Visualization
![post-training-visualization](/output2.png)
![post-training-visualization](/output3.png)
- Prediction (New Input)

## Tech Stack and Tools
- Programming language
    - Python 
- libraries
    - scikit-learn
    - pandas
    - numpy
    - seaborn
    - matplotlib
- Environment
    - Jupyter Notebook
    - Anaconda
- IDE
    - VSCode

You can install all dependencies via:
```
pip install -r requirements.txt
```



## Usage Instructions
To run this project locally:
1. Clone the repository:
```
git clone https://github.com/charlesakinnurun/shell-stock.git
cd shell-stock
```
2. Install required packages
```
pip install -r requirements.txt
```
3. Open the notebook:
```
jupyter notebook model.ipynb

```

## Project Structure
```
customer-personality/
│
├── model.ipynb  
|── model.py    
|── shell_stock_data.csv  
├── requirements.txt 
├── shell.jpg       
├── output1.png        
├── output2.png  
├── output3.png      
├── SECURITY.md        
├── CONTRIBUTING.md    
├── CODE_OF_CONDUCT.md 
├── LICENSE
└── README.md          

```
