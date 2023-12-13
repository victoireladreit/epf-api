# EPF-API-TP

- **Question 1:** _Which Python library/framework is often used to create fast, simple REST APIs?_

  - Django

  - ***Flask***

  - FastAPI

  - All of the above


- **Question 2:** _What's the main difference between Django, Flask and FastAPI in terms of performance and speed?_

  - Django is generally faster than Flask and FastAPI.

  - Flask outperforms Django and FastAPI.

  - **_FastAPI is renowned for its increased speed and performance compared with Django and Flask._**

  - Django, Flask and FastAPI have equivalent performance.


- **Question 3:** What is an endpoint in the context of REST APIs?\*

  - A unique IP address associated with an API.

  - A breakpoint in the code where the API can be interrupted.

  - **_A specific URL to which a request can be sent to interact with the API._**

  - A unique identifier assigned to each incoming request.


- **Question 4:** _What are the main HTTP verbs used to define REST API methods?_

  - **_GET, POST, PUT, PATCH, DELETE_**

  - SEND, RECEIVE, UPDATE, REMOVE

  - READ, WRITE, MODIFY, DELETE

  - FETCH, INSERT, UPDATE, DELETE


- **Question 5:** _In the context of REST APIs, what does the term "middleware" mean?_

  - A component that processes data sent by the user.

  - An external library used to speed up API development.

  - **_Intermediate software that processes the request before it reaches the main application._**

  - A method for securing data stored in the database.


- **Question 6:** _Which Python library is often used to serialize and deserialize JSON data in the context of REST APIs?_

  - JSONify

  - PyJSON

  - **_json.dumps() and json.loads()_**

  - serializeJSON


- **Question 7:** _What is the main use of the HTTP "PUT" method in the context of REST APIs?_

  - Create a new resource.

  - **_Update an existing resource, or create one if it doesn't exist._**

  - Delete a resource.

  - Read a specific resource.


- **Question 8:** In FastAPI, how do you define an endpoint to handle a POST request with JSON data?\*

  - **_@app.post("/endpoint")_**

  - @app.get("/endpoint")

  - @app.request("/endpoint")

  - @app.update("/endpoint")


# Creating an API with FastAPI

### Introduction

In this exercise you are going to encapsulate data processing and machine learning model execution logic to make predictions on a well-known dataset: iris kaggle dataset

### Few elements to remember about the REST Protocol

REST (Representational State Transfer) is an architectural style for designing networked applications. RESTful APIs (Application Programming Interfaces) conform to the principles of REST, allowing systems to communicate over HTTP in a stateless manner; Some important aspects are:

- **Resources:** Everything is a resource, identified by a unique URI.

- **HTTP Methods:** CRUD operations are performed using standard HTTP methods (GET, POST, PUT, DELETE).

- **Stateless:** Each request from a client contains all the information needed to understand and fulfill the request.

### Key Concepts in FastAPI:

- **Endpoint:**

- **Basic HTTP Methods:**

- **Request and Response:**

### Objective

Before starting the exercise, download the contents of the main branch of the following git repo and create your own private repo: https://github.com/klemmm922/EPF-API-TP/ \
Don't forget to add me to the repo with the following email address: clement.letizia@epfedu.fr


- **Step 1: Installing libraries:** Install the libraries in the requirements.txt

- **Step 2: First launch:**  Execute the main.py file in the root folder and access it.

- **Step 3: Redirect root API:**  Redirect the root endpoint of your API to the automatic swagger documentation

- **Step 4: Access the swagger documentation:**  Access to the swagger built automatically by FastAPI

- **Step 5: First call to the API:**  Make an API request on the hello route using the swagger directly or a tool like insomnia or postman

- **Step 6: Access the dataset:**  Create a route in api/routes/data to download and save the contents of the following kaggle dataset in the src/data folder: https://www.kaggle.com/datasets/uciml/iris. If you're having too many problems, simply download the dataset from the kaggle website. documentation : https://www.geeksforgeeks.org/how-to-download-kaggle-datasets-into-jupyter-notebook/

- **Step 7: Loading the Iris Flower dataset:** Add an endpoint to load the iris dataset file as a dataframe and return it as a json.

- **Step 8: Processing the dataset:** Add an endpoint to be able to perform the necessary processing on the data before being able to train a model with it.

- **Step 9: Split in train and test:** Add an endpoint to split the iris dataset as train and test and send back a json with both

- **Step 10: Parameters init:** Go to scikit learn and select any classification model to be used on the iris dataset (performance is of no interest to us in this course). Look at the parameters you need to use for this model and store them in the file src/config/model_parameters.json

- **Step 11: Training the classification model:** Add an endpoint to train a classification model with the processed dataset as input and saved this model in the folder src/models.

- **Step 12: Prediction with Trained Model:** Add endpoint to make predictions with trained model and parameters. This endpoint have to send back the predictions as json.

- **Step 13: Create the Firestore collection:** Create the firestore collection "parameters" with the following parameters: "n_estimators", "criterion".

- **Step 14: Retrieve parameters from Firestore:** Add an endpoint to retrieve parameters from Firestore.

- **Step 15: Update and add Firestore parameters:** Add endpoints to update or add parameters in Firestore.

### Evaluation requirements

The evaluation criteria will be as follows:

- Proper functioning of endpoints
- Clear documentation of code, use of explicit names and compliance with REST naming conventions. Please follow pep-8 convention for documenting your functions (exemple at the end of the README)
- Static swagger generation through an API route
- Completion of unit tests

The completion of this TP is relatively long and may overtake TP3 

### Documentation link :

- FastApi: https://fastapi.tiangolo.com/

- Google Cloud Firestore: https://cloud.google.com/python/docs/reference/firestore/latest/index.html

- Scikit-Learn: https://scikit-learn.org/stable/index.html

- Pandas: https://pandas.pydata.org/docs/


### Pep-8 docstring example :

"""
  Retrieve content of a json file

  Args:
      path (str): The path of the file

  Returns:
      JSON object: The json object with the parameters
  """
