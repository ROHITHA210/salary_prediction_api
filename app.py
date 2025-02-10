#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, request, jsonify
import psycopg2

app = Flask(__name__)

# Replace with your Railway PostgreSQL Connection URL
DB_URL = "postgresql://postgres:IfYbzdzkSkSWKJaxuPsryDzpYFLeKlQR@roundhouse.proxy.rlwy.net:39423/railway"

# Route to check if API is running
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Salary Prediction API is running!"})

# Route to save predictions
@app.route("/predict", methods=["POST"])
def save_prediction():
    try:
        data = request.get_json()
        role = data["role"]
        company = data["company"]
        location = data["location"]
        experience = float(data["experience"])
        company_size = data["company_size"]
        skills = data["skills"]
        predicted_salary = float(data["predicted_salary"])

        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()

        # Insert prediction into PostgreSQL
        cur.execute("""
            INSERT INTO salary_predictions (role, company, location, experience, company_size, skills, predicted_salary)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (role, company, location, experience, company_size, skills, predicted_salary))

        conn.commit()
        cur.close()
        conn.close()

        return jsonify({"message": "✅ Prediction saved!"})

    except Exception as e:
        return jsonify({"error": str(e)})

# Route to retrieve saved predictions
@app.route("/predictions", methods=["GET"])
def get_predictions():
    try:
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()

        cur.execute("SELECT * FROM salary_predictions;")
        rows = cur.fetchall()

        cur.close()
        conn.close()

        predictions = [
            {"id": row[0], "role": row[1], "company": row[2], "location": row[3], 
             "experience": row[4], "company_size": row[5], "skills": row[6], "predicted_salary": row[7]}
            for row in rows
        ]

        return jsonify(predictions)

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
# Run the Flask app
#if __name__ == "__main__":
    #app.run(debug=True)


# In[2]:


#get_ipython().system('pip install flask flask-cors psycopg2-binary')


# In[3]:


from flask import Flask, request, jsonify
import psycopg2

app = Flask(__name__)

# Replace with your Railway PostgreSQL Connection URL
DB_URL = "postgresql://postgres:IfYbzdzkSkSWKJaxuPsryDzpYFLeKlQR@roundhouse.proxy.rlwy.net:39423/railway"

# Route to check if API is running
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Salary Prediction API is running!"})

# Route to save predictions
@app.route("/predict", methods=["POST"])
def save_prediction():
    try:
        data = request.get_json()
        role = data["role"]
        company = data["company"]
        location = data["location"]
        experience = float(data["experience"])
        company_size = data["company_size"]
        skills = data["skills"]
        predicted_salary = float(data["predicted_salary"])

        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()

        # Insert prediction into PostgreSQL
        cur.execute("""
            INSERT INTO salary_predictions (role, company, location, experience, company_size, skills, predicted_salary)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (role, company, location, experience, company_size, skills, predicted_salary))

        conn.commit()
        cur.close()
        conn.close()

        return jsonify({"message": "✅ Prediction saved!"})

    except Exception as e:
        return jsonify({"error": str(e)})

# Route to retrieve saved predictions
@app.route("/predictions", methods=["GET"])
def get_predictions():
    try:
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()

        cur.execute("SELECT * FROM salary_predictions;")
        rows = cur.fetchall()

        cur.close()
        conn.close()

        predictions = [
            {"id": row[0], "role": row[1], "company": row[2], "location": row[3], 
             "experience": row[4], "company_size": row[5], "skills": row[6], "predicted_salary": row[7]}
            for row in rows
        ]

        return jsonify(predictions)

    except Exception as e:
        return jsonify({"error": str(e)})

# Run Flask inside Jupyter Notebook
from werkzeug.serving import run_simple
import threading

def run_app():
    run_simple('localhost', 5000, app)

threading.Thread(target=run_app).start()


# In[4]:


#lets try and save a salary prediction
import requests

url = "http://127.0.0.1:5000/predict"
data = {
    "role": "Data Scientist",
    "company": "Google",
    "location": "Bangalore",
    "experience": 5,
    "company_size": "1000+",
    "skills": "Python, SQL, ML",
    "predicted_salary": 1500000
}

response = requests.post(url, json=data)
print(response.json())


# In[5]:


#to retrieve allpredictions
response = requests.get("http://127.0.0.1:5000/predictions")
print(response.json())


# In[6]:


get_ipython().system('jupyter nbconvert --to script your_notebook_name.ipynb')


# In[ ]:




