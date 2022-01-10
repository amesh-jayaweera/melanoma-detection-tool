## Melanoma Detection Tool : REST API

#### Technologies & Libraries Used
    Python 3.9
    Flask framework 
    scikit-learn==1.0
    numpy
    pandas
    gunicorn
    flask_cors
    
### How to setup and run
##### Create virtual environment 
###### Windows 
    py -3 -m venv <name of environment>
###### Linux/MaxOS
    python3 -m venv <name of environment>
##### Activate virtual environment 
###### Windows 
    <name of environment>\Scripts\activate
###### Linux/MaxOS
    . <name of environment>/bin/activate
##### Install required libraries
    pip3 install -r requirements.txt
##### Run app locally
    flask run

#### Special Notes
* Use Python 3.9 and scikit-learn==1.0
* If any new library requires to install, after install freeze it to the requirements.txt
* add your virtual environment directory to .gitignore 