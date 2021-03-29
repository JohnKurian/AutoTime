== MLFlow == 

- Download source from github and replace python library package with source
- Then, mlflow server
mlflow server --backend-store-uri sqlite:///mlruns.db --default-artifact-root ./mlruns

- Then npm install to run UI.
- To run the UI, in mlflow/server/js, npm start --host 0.0.0.0

Creating virtualenv
python -m venv autotime_env

SSL
pip --proxy http://geo-cluster125184-swg.ibosscloud.com:8082 --cert ca-bundle-with-iboss.pem install streamlit --user


Starting the flask backend API for frontend 
flask run --with-threads

Starting Autotime frontend
cd flask-react-app
npm install 
npm start  


Ports: 
mlflow UI == 3000
mlflow server == 5000
flask server == 8000
autotime frontend == 4000
model server == 8080