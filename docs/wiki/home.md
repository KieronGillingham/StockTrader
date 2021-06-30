# Stock Trader
The Stock Trader is a machine learning application written in Python and using scikit-learn to forecast future values of stock prices.
It was created as part of a 2020 machine learning course at Solent University.

# Contents
1. Overview
2. Set-up
3. User Guide
4. Design and Implementation

# Overview
The Stock Trader is a Python application packaged as a Windows executable. It uses a Multilayer Perceptron model to analyse stock data from Yahoo Finance and provide predictions of stock prices on specified future dates. This prediction information is displayed in a chart.

# Set-up
TBC

# User Guide
## User Management
A simple user managing system is included in the application.
---
*WARNING* This implementation is unsecured and should not be used in a production environment. Do not store sensitive details in this system.
---
![Use case diagram](https://github.com/KieronGillingham/StockTrader/blob/main/docs/UseCases.png)
### Login
When starting the application, the user is prompted to sign in. Enter valid details for a registered user. The example accounts are included in the file [users.csv](../../data/users.csv). A new account can also be created by clicking 'Register'.
Alternatively, select 'Continue as Guest' to sign in using a guest account. The guest account has no funds by default. For [investment suggestions](#InvestmentSuggestions) you must first add funds through the [user page](#UserPage).

![Login page](images/loginpage.png)

### Registration
Through the registration page, new a new account can be created. Fill in the form and select 'Register'. New accounts have a default balance of £0. This can be changed from the [user page](#UserPage) after signing in.

![Registration page](images/registerpage.png)

A new user cannot share a username with an existing user. There are no size or character requirements for passwords and they are stored in an unsecured plaintext file.

### User Page
After signing-in to the application, the user can navigate to the user page. From here the user can update their username and password, or add funds to their account.

![User page](images/userpage.png)


## Chart
The chart page is the main view of the application. It features a large canvas for displaying stock prices overtime and predictions for future values.
![User page](images/userpage.png)

## Loading Data
Before the application can perform any machine learning or forecasting, stock data must be loaded. The application is designed to load this data from either Yahoo Finance or a local CSV file.
### Yahoo Finance
TBC

### Local CSV File
TBC

## Forecasting
TBC

## Investment Suggestions
TBC

# Design and Implementation
TBC

## Machine Learning

## Use Cases
![Use case diagram](https://github.com/KieronGillingham/StockTrader/blob/main/docs/UseCases.png)

## Wireframes
![User interface wireframe diagram](https://github.com/KieronGillingham/StockTrader/blob/main/docs/UIWireframes.png)

## Data Flow
![User interface wireframe diagram](https://github.com/KieronGillingham/StockTrader/blob/main/docs/UIWireframes.png)

## Class Diagram
![Class diagram](https://github.com/KieronGillingham/StockTrader/blob/main/docs/UIWireframes.png)

## Technologies and Libraries used
- Python
- scikit-learn
- yahoofinancials
- matplotlib
- PyQt5