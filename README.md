# Project_telco_churn

## Overview

This is a Machine Learning school project with a full workflow,
database and StreamLit integration to predict the customer churn 
rate for the dataset *Telco Customer churn* created by **BLASTCHAR** over at Kaggle. 

*Link to dataset* - https://www.kaggle.com/datasets/blastchar/telco-customer-churn 

## How to run

In order to run the software, you'll need to have python installed and you'll also need our .env-file which isn't 
provided in the repo by default. Butif you've been granted access to it, make sure to paste it into the root of the project.

**Setup**

* Open the file
* Create a .venv

Once that it complete, follow these next steps

* Open your command terminal
* Run "pip install requirements.txt"

Great, onto the actual program. 

**Running the program**

1. Open your terminal and paste the command below into it:
 * uvicorn serving.src.telco_api.main:app --reload --port 8000

*(This loads loads spins up our Machine Learning workflow)*

2. Open another terminal and paste the command below into it:
* streamlit run frontend/app.py

3. Click the link shown in the command prompt if it doesn't
open your webbrowser by default. 

Congrats! You'll all up and running now!

**How to use**

It's very user-friendly, simply enter the variables of your customer and hit the red button saying "*Predict Churn Risk*".

### Extra feature

If you'd like a sneak peak underneath the hood, so to speak, to see how it works. 
You can, with a new terminal, use the command: 
* *python -m cli.cli*

This is where we train the models themselves.

## Contributors 

* Pyr0xd - Timoty Bengtsson
* OscarK99Swe - Oscar Kock
* M-Renberg - Mikael Renberg