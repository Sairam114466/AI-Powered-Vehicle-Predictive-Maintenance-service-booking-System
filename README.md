# AI-Powered Vehicle Predictive Maintenance System

## Project Overview

This project presents an intelligent vehicle predictive maintenance system that combines artificial intelligence, machine learning, and a user-friendly web interface to detect early signs of vehicle problems before they become severe. Most vehicle owners rely on fixed service intervals or wait until a warning light appears, which often means the issue has already escalated. The proposed system addresses this challenge by predicting probable maintenance issues using vehicle operating data such as engine speed, oil pressure, fuel pressure, coolant temperature, and engine temperature difference.

The solution is designed as a final-year academic project with practical relevance. It uses machine learning algorithms to classify vehicle condition and highlight maintenance risk, while also presenting a simple decision-support dashboard for users. The web application allows users to input vehicle parameters, evaluate the current health condition, and receive actionable recommendations for preventive maintenance.

## Problem Statement

Traditional vehicle maintenance is often reactive or scheduled without considering the current health of the vehicle. This can result in unnecessary service costs, missed early warning signs, and serious mechanical failures. There is a strong need for a smart maintenance system that can analyse vehicle condition and predict possible faults before they lead to breakdowns.

## Objectives

- To develop a predictive maintenance system for vehicles using AI and machine learning.
- To analyse vehicle health indicators and identify possible maintenance risks.
- To build a user-friendly web dashboard for vehicle condition prediction.
- To support proactive maintenance decisions before major vehicle failures occur.
- To present a final-year project that demonstrates intelligent decision-making and practical machine learning application.

## System Features

- AI-based prediction of vehicle condition using machine learning
- Real-time dashboard for entering vehicle parameters
- Engine health analysis based on operating sensor values
- Risk identification for maintenance attention
- User-friendly interface for final-year project demonstration
- Service recommendation guidance for preventive maintenance

## Technologies Used

- Python
- Streamlit
- Pandas
- Scikit-learn
- NumPy
- Machine learning model: Gradient Boosting Classifier

## Dataset

The project uses the engine data file stored in the repository. The dataset includes vehicle operating parameters and engine condition labels used for training and prediction.

## Project Workflow

1. Load the vehicle dataset.
2. Pre-process the relevant engine variables.
3. Compute temperature difference using the available features.
4. Train a machine learning model for classification.
5. Predict vehicle condition based on user-entered values.
6. Display result, probability, and maintenance recommendation on the dashboard.

## Abstract

Vehicle breakdowns rarely happen without warning signs, but most owners do not have an easy way to detect these signs before it is too late. Traditional maintenance methods either repair a vehicle only after failure or service it based on fixed intervals without considering the actual condition of the vehicle. Both approaches are inefficient, resulting in emergency repair costs, safety risks, unnecessary servicing, and missed maintenance opportunities. This project proposes an AI-based predictive maintenance system that uses machine learning to identify potential vehicle faults in advance based on parameters such as mileage, engine temperature, oil pressure, fuel pressure, and coolant condition.

The system does not require expensive IoT sensors or physical equipment. All required input data can be entered manually through a web-based interface, making the solution affordable, simple, and practical. The core of the system is a machine learning model trained to classify engine condition and estimate the likelihood of a fault using algorithms such as Gradient Boosting. Several model evaluations are performed to ensure reliability and stability.

To make the system useful in real life, the web application presents vehicle health predictions in a clear dashboard and recommends maintenance actions when risk is detected. This project demonstrates how artificial intelligence and machine learning can be used to move from reactive vehicle maintenance to proactive and intelligent service planning. The result is a practical final-year project that combines predictive analytics, user interaction, and smart vehicle maintenance support in one complete system.

## Introduction

Cars and bikes are an essential part of daily life, whether for commuting, business, or transportation. However, many vehicle owners do not know the true operating condition of their vehicle until a visible problem or warning signal appears. In most cases, the warning comes too late, after the issue has already increased in severity and cost. Existing maintenance practices usually rely on either fixed schedule servicing or repair after a breakdown, both of which ignore the actual health of the vehicle.

This creates unnecessary maintenance costs on one side and potentially dangerous vehicle failures on the other. A more effective approach is to detect performance issues early and act before they become critical. Machine learning offers this capability by learning patterns from past vehicle behavior and identifying signs that may indicate future faults.

This project develops an AI-based predictive maintenance system for vehicles. The system takes key operating parameters as inputs, predicts whether the vehicle is in a healthy or risky condition, and helps users make proactive maintenance decisions. By integrating machine learning with a web dashboard, the project provides a practical and understandable solution for real-world use.

## Run the Project

1. Open the project folder.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```
4. Open the local URL shown in the terminal.

## MySQL Storage

The app uses SQLite by default. To use a local MySQL 8 server, install the requirements and set these variables in PowerShell before starting the app:

```powershell
$env:DB_BACKEND = 'mysql'
$env:MYSQL_HOST = '127.0.0.1'
$env:MYSQL_PORT = '3306'
$env:MYSQL_USER = 'root'
$env:MYSQL_DATABASE = 'vehicle_app'
$securePassword = Read-Host 'Enter MySQL password' -AsSecureString
$env:MYSQL_PASSWORD = [System.Net.NetworkCredential]::new('', $securePassword).Password
streamlit run app.py
```

The app creates the `vehicle_app` database and its tables, then copies existing users and bookings from `vehicle_app.db` once. The MySQL password is read from the environment and is not stored in the source code.

To view the app data in the MySQL command line, connect with `mysql -u root -p` and run:

```sql
USE vehicle_app;
SHOW TABLES;
SELECT id, name, email, role FROM users;
SELECT id, user_email, service_type, scheduled_for, status FROM bookings;
```

## Contributors

- Dharshan A
- Pravin R F
- Rishikaandh Devadoss
- Abinayaa M K

## Contact

For questions or project-related queries, contact the project team or academic supervisor.
