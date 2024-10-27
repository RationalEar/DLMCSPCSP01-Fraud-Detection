## DLMCSPCSP01 AI-Based Fraud Detection

This project aims to investigate and implement various AI techniques for detecting fraudulent 
financial transactions. The primary focus will be on analyzing transaction data to identify anomalies 
indicative of fraud using machine learning algorithms. The project will involve the collection and preprocessing of 
financial transaction data, followed by the extraction of key features such as transaction amount, frequency, 
geographical patterns, and user behavior. A range of machine learning techniques, including supervised learning 
methods like Logistic Regression and Random Forests will be explored to enhance detection accuracy.

### Getting Started
1. Start the server by running the following command:
```
python main.py
```
2. Open the browser and navigate to the following URL to generate training and testing data:
```
http://localhost:5000/generate-data
```
3. Train the model by navigating to the following URL:
```
http://localhost:5000/train
```
4. Run real time detection by navigating to the following URL:
```
http://localhost:5000/predict
```