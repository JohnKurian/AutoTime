## BOAT

Source code for the 2021 IEEE BigDataService paper ["BOAT: A Bayesian Optimization AutoML Time-series Framework for Industrial Applications"](https://github.com/JohnKurian/BOAT/blob/main/BOAT.pdf).

## Setting up MLFlow backend

- Download source from github with the folder name "mlflow" at root location
- Start the server with the following command
```
mlflow server --backend-store-uri sqlite:///mlruns.db --default-artifact-root ./mlruns
```
- To run the MLFlow UI, in mlflow/server/js, run the following command
```
npm start --host 0.0.0.0
```

## Starting the backend
```
flask run --with-threads
```

## Starting BOAT frontend
```
cd flask-react-app
npm start  
```

## Ports
mlflow UI --> 3000
mlflow server --> 5000
flask server --> 8000
BOAT frontend --> 4000
model server --> 8080
