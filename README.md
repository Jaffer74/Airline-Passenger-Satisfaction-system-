# Airline Passenger Satisfaction System

A machine learning project aimed at enhancing the airline passenger experience by predicting satisfaction levels using a range of supervised and unsupervised models. This project implements algorithms to analyze passenger feedback, optimize operations, and uncover insights for improving customer satisfaction in the airline industry.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Dataset](#dataset)
4. [Models Used](#models-used)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Results](#results)
8. [Contributing](#contributing)
9. [License](#license)

## Project Overview
Airline customer satisfaction is vital for competitive performance and customer loyalty in the airline industry. This project analyzes airline passenger data using machine learning techniques to understand factors impacting passenger satisfaction and classify satisfaction levels. It uses a combination of supervised models like Logistic Regression, Naive Bayes, Decision Trees, and Random Forest, as well as unsupervised methods like K-Means and DBSCAN for clustering.

## Features
- **Supervised Models:** Predict customer satisfaction using classifiers such as Logistic Regression, Naive Bayes, K-Nearest Neighbors (KNN), Decision Trees, Random Forest, and Support Vector Machine (SVM).
- **Unsupervised Models:** Identify patterns in customer feedback using clustering techniques like K-Means and DBSCAN.
- **Data Preprocessing:** Handles missing values, outliers, feature selection, and encoding.
- **Feature Selection:** Uses Chi-Square tests and Wrapper methods for relevant feature selection.
- **Evaluation Metrics:** Accuracy, ROC-AUC score, confusion matrix, and visualizations.

## Dataset
The dataset includes details of 103,904 airline passengers, with features like:
- **Demographics:** Age, Gender, Customer Type
- **Flight Details:** Flight Distance, Type of Travel, Class
- **Service Ratings:** In-flight wifi, Seat comfort, Cleanliness, Food and beverages, On-board service, etc.
- **Outcome:** Passenger satisfaction labeled as "Neutral or Dissatisfied" or "Satisfied"

## Models Used
1. **Logistic Regression**
2. **Naive Bayes**
3. **K-Nearest Neighbors (KNN)**
4. **Decision Tree**
5. **Random Forest**
6. **Neural Network (MLP)**
7. **Support Vector Machine (SVM)**
8. **XGBoost**
9. **AdaBoost**
10. **K-Means Clustering**
11. **DBSCAN**

## Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/airline-passenger-satisfaction.git
   cd airline-passenger-satisfaction
   ```

2. **Install the dependencies:**
   Make sure you have `pip` installed and use the following command:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. **Preprocess the Data and Run Models:**
   The main functionality of the project, including data preprocessing, training models, and evaluation, is in `Flight_Passenger_Satisfaction.py`. To execute the script, use:
   ```bash
   python Flight_Passenger_Satisfaction.py
   ```

2. **Script Sections:**
   - **Data Preprocessing:** Cleans and transforms the data, including handling missing values, encoding categorical features, and scaling.
   - **Model Training:** Trains various models, including Logistic Regression, Decision Trees, and Random Forests, and outputs performance metrics like accuracy and ROC-AUC.
   - **Clustering Analysis:** Performs clustering using K-Means and DBSCAN to identify patterns among different passenger groups.
   - **Evaluation:** Displays classification reports and confusion matrices for each supervised model, as well as ROC curves to assess performance.

3. **Interpreting Results:**
   The script will output accuracy scores, ROC-AUC, and confusion matrices for each supervised model, providing insights into the factors influencing customer satisfaction.

## Results
- **Best Model:** Random Forest achieved the highest accuracy after hyperparameter tuning (96% accuracy).
- **Clustering:** K-Means and DBSCAN were used to uncover clusters based on customer demographics and service ratings, although clustering accuracy was moderate.
- **Performance Metrics:** Each model’s performance is evaluated using accuracy, ROC-AUC, and training time.

## Documentation
For detailed methodology, results, and analysis, see the [IEEE Report](report.pdf).

## Contributing
We welcome contributions! Please see `CONTRIBUTING.md` for guidelines.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
