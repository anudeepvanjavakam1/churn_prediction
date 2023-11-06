# churn prediction
A flask app to predict customer churn for a subscription service business

## Problem Description
- Customer churn (customers who cancel their subscriptions) is a vital metric for businesses offering subscription services.
- Predictive analytics techniques are employed to anticipate which customers are likely to churn, enabling companies to take proactive measures for customer retention.
- This project focuses on predicting the probability of customer churn

## Installation

1. git clone https://github.com/anudeepvanjavakam1/churn_prediction.git
2. cd into the directory
3. In terminal, run command 'pipenv install'

Pipenv will read the Pipfile and Pipfile.lock files for the project, create the virtual environment, and install all of the dependencies as needed.
To activate the environment, just navigate to your project directory and use pipenv shell to launch a new shell session, or use pipenv run <command> to run a command directly.

You can use the Dockerfile to build an image and run it with the following commands:
1. docker build -t python-imagename .
2. docker run python-imagename

## Files

- train and test data sets along with data descriptions file are in the data folder
- framework_setup.ipynb jupyter notebook has EDA, multiple modeling approaches, feature importances, rationale for evaluation metric, hyperparameter tuning.
- train.py has the logic for training the model and exports the model and DictVectorizer (churn_xgb_model.sav and dv.sav) required for predictions on the test set.
- predict.py has the code to run Flask app
- testing_flask.ipynb jupyter notebook has an example of how to pass test examples to the app for predictions

## Results
- This data is highly imbalanced and hence precision and recall are chosen as metrics with undersampling the majority class. The decision threshold (0.76) for which there's a max f1 score is chosen to classify churn or no churn
- No significant multicollinearity among features is found
- Churn probability increases with account age
- The higher the Average Viewing Duration, content downloads per month, and viewing hours per week, the lower the churn probability
- The higher the monthly charges and support tickets per month, the higher the churn probability
- Subscription Type (Basic, standard, premium), Payment method, User rating, Paperless Billing, Genre preference, and watchlist size did not have a significant influence on churn

## Improvements
- Scope for Feature Engineering to improve f1_macro score
- Trying out other under and over-sampling techniques as well as resampling with different ratios
- Expand the Flask app for batch predictions
- Trying other cost sensitive learning algorithms to penalize classification mistakes in minority class
- If time permits, exhaustively grid searching for hyperparameter tuning optimization
- Deploying to the cloud with an easy to use UI
