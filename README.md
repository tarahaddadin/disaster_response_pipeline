# Disaster Response Pipeline Project

This project is part of the Data Science Nanodegree Program by Udacity in collaboration with Figure Eight.

An API that uses machine learning to categorise messages received during a crisis. Training a machine learning model on 30,000 real messages received during events such as the 2010 earthquakes in Haiti and Chile, floods in Pakistan in 2010 and super-storm Sandy in 2012, the API helps organisations focus on getting help to the right places.

This project is divided into three key sections:
1. Processing the data, building an ETL pipeline to extract data from source, clean the data and save it into a SQLite database.
2. Build a machine learning pipeline to train the data
3. Run a web app in Flask, which shows the model results in real time

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://127.0.0.1:5000/

## Authors 
* [Tara Haddadin](https://github.com/tarahaddadin)

## Acknowledgements
* [Udacity](https://www.udacity.com/) for providing an amazing Data Science Nanodegree Program
* [Figure Eight](https://www.figure-eight.com/) for providing the relevant dataset to train the model

