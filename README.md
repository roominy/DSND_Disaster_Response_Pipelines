# Disaster Response Pipeline Project
The Disaster Response Pipelines Project is part of Udacity's Data Science Nano Degree Program. 

### Table of Contents
1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Instructions](#instructions)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## 1. Installation <a name="installation"></a>

To be able to run and view this project. It's recommended to have the latest versions of the followings:
* [Python 3](https://www.python.org/downloads/)
* [Pandas](https://pandas.pydata.org)
* [NumPy](https://numpy.org/)
* [plotly](https://pypi.org/project/plotly/)
* [flask](https://pypi.org/project/Flask/)
* [nlkt](https://pypi.org/project/nltk/)
* [joblib](https://pypi.org/project/joblib/)
* [sqlalchemy](https://pypi.org/project/SQLAlchemy/)
* [sklearn](https://pypi.org/project/sklearn/)

## 2. Project Motivation <a name="motivation"></a>

 In this project, I appled data engineering techineqws analyzed disaster data from Figure Eight to build a model for an API that classifies disaster messages. 
 
## 3. File Descriptions <a name="files"></a>

The project consist from three main folders:

    - data folder that contains:
        - disaster_messages.csv: contains the dataset that includes the messages.
        - disaster_categories.csv: contains the dataset that includes messages categories.
        - process_data.py: ETL pipeline script that reads the datasets, merges the two datasets and cleans the data, then saves the dataset into a database file.
        - DisasterResponse.db: the outcome of the ETL pipeline (SQLite database containing a table that merges the messages and categories data).

    - model folder that contains:
        - train_classifier.py: machine learning pipeline script that loads data from the SQLite database, splits the dataset into training and test sets,  process text and train test the classifier the export the trained  classifier into pkl file. 
        - classifier.pkl: the outcome of the machine learning pipeline ( the trained classifer).

    - app folder that contains:
        - run.py: script to run the Flask web app
        - templates folder: contains html files of the web app.

 


## 4. Instructions <a name="instructions"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## 5. Licensing, Authors, and Acknowledgements <a name="licensing"></a>

Credit given to Udacity courses for code ideas and motivation , and to figure 8 for the data.

Author: NYRoomi


