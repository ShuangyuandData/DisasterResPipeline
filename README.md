# DisasterResPipeline

Libraries: None

### Summary:
This project analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.
I build a web app where an emergency worker can input a new message and get classification results in 36 categories.
The detailed methods include the ETL process, TF_IDF pipeline, randomforest classifier, and grid search.

Motivation: Analyze a data set containing real messages that were sent during disaster events and create a machine learning pipeline to categorize these events so that the messages can be sent to an appropriate disaster relief agency.

Descriptions of files:
    1. data directory: the files related with data and ETL process
    2. model directory: the files related with the classifier modeling
    3. preparation directory: the files related with the pre-process
    4. app directory: the files related with the web deployment

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/ (env | grep WORK, https://SPACEID-3001.SPACEDOMAIN)

