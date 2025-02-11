# salary_prediction_api
#It is the instructions file for my project.
                                             🔹 Salary Prediction API – Project Summary
                                            
Purpose – This API predicts salaries based on job details such as role, company, experience, and skills.
Tech Stack Used – Python, Flask, PostgreSQL, Docker, Railway.
Machine Learning – Uses Supervised Learning & K-Means Clustering for Predictions.
Database – Stores predictions in PostgreSQL on Railway.app for future reference.
Deployment – Deployed as a Flask API using Docker & Railway.app.

🎯Project Overview: This project is a Salary Prediction system that uses Supervised Machine Learning and K-means clustering to predict salaries based on job roles, experience, company size, and skills. The predictions are stored in a PostgreSQL database and deployed online using Flask and Railway.app.

🎯Problem Statement: 
  Job seekers often want to estimate salaries before applying for jobs.
  The goal of this project is to predict salaries using Machine Learning models.
  The predictions are stored in a database for future reference.

✨Dataset Description:
    Role                                 – Job title (e.g., Data Scientist, Software Engineer).
    Company                              – The company offering the job.
    Location                             – Job location.
    Experience                           – Required experience in years.
    Company Size                         – Size of the company (e.g., 1000+ employees).
    Skills                               – Required skills (e.g., Python, SQL, Machine Learning).
    Min Salary, Max Salary, Salary       – Salary details.

✨Data Preprocessing:
  Before training the model, the dataset underwent the following preprocessing steps:
    Handling Missing Values     – Missing salary and company size values were handled using median imputation.
    Feature Engineering         – Columns with spaces were renamed (e.g., "Company Size" → "Company_Size").
    Scaling & Normalization     – Min Salary, Max Salary, and Experience were normalized using StandardScaler.
    Encoding Categorical Data   – Role, Company, Location, and Skills were encoded using OneHotEncoder.

✨Model Training:
The model uses Supervised Learning & K-Means Clustering for salary predictions:
  Supervised Learning Model     – Linear Regression was trained to predict salaries based on job details.
  K-Means Clustering            – Used to group job roles into similar salary categories.
  Model Evaluation              – RMSE (Root Mean Square Error) was used to assess accuracy.

🎯API Endpoints:
To check if API is running, copy and paste the URL in any browser: https://salarypredictionapi-production.up.railway.app/
The response should look like: {"message": "Salary Prediction API is running!"}

A list of all stored predictions in JSON format can be seen through: https://salarypredictionapi-production.up.railway.app/predictions
The response(sample) looks like:
[{"company":"Google","company_size":"1000+","experience":5.0,"id":1,"location":"Bangalore","predicted_salary":1500000.0,"role":"Data Scientist","skills":"Python, SQL, ML"},{"company":"Google","company_size":"1000+","experience":5.0,"id":2,"location":"Bangalore","predicted_salary":1500000.0,"role":"Data Scientist","skills":"Python, SQL, ML"},{"company":"Google","company_size":"1000+","experience":5.0,"id":3,"location":"Bangalore","predicted_salary":1500000.0,"role":"Data Scientist","skills":"Python, SQL, ML"},{"company":"Google","company_size":"1000+","experience":5.0,"id":4,"location":"Bangalore","predicted_salary":1500000.0,"role":"Data Scientist","skills":"Python, SQL, ML"},{"company":"Microsoft","company_size":"5000+","experience":3.0,"id":5,"location":"Hyderabad","predicted_salary":0.0,"role":"Software Engineer","skills":"Java, Spring Boot, SQL"}]

✨Deployment Steps: This project is deployed using Docker & Railway.app
  1️⃣ Set up PostgreSQL             – Connected to a PostgreSQL database on Railway.app.
  2️⃣ Create a Flask API            – Developed API endpoints for salary predictions.
  3️⃣ Containerization with Docker  – Dockerized the Flask app for smooth deployment.
  4️⃣ Deploy on Railway.app         – Hosted the project online for real-world use.
To use the API locally, clone the repository: git clone https://github.com/your-username/salary-prediction-api.git
                                              cd salary-prediction-api
Next run the following commands- pip install -r requirements.txt
                                 DATABASE_URL=your_postgresql_connection_url
                                 python app.py

Future Enhancements
✔ Add more machine learning models (Random Forest, XGBoost).
✔ Improve dataset quality & add salary benchmarks from real-world data.
✔ Build a front-end UI for better user experience.
Now let us understand the code:

Step 1: Load all necessary libraries for data handling, preprocessing, ML training, and PostgreSQL integration.
# Import necessary libraries
  !pip install flask pandas numpy scikit-learn joblib psycopg2 gunicorn  #flask → For building the web application
                                                                         #pandas, numpy → For data processing
                                                                         #scikit-learn → For machine learning models
                                                                         #joblib → For saving/loading models
                                                                         #psycopg2 → For connecting to PostgreSQL
                                                                         #gunicorn → For deploying on railway
  import matplotlib.pyplot as plt                               # For visualization
  import seaborn as sns                                         # For advanced visualization
  from sklearn.model_selection import train_test_split          # Splitting dataset
  from sklearn.preprocessing import StandardScaler              # Scaling numerical data
  from sklearn.linear_model import LinearRegression             # Regression model
  from sklearn.cluster import KMeans                            # Clustering algorithm
  from sklearn.impute import SimpleImputer
  
Step 2: Load the dataset into a Pandas DataFrame and check the structure.
# Upload the dataset
  df = pd.read_csv("salary_dataset.csv")  # Replace with your actual dataset filename
# Display the first few rows
  df.head()
  df.columns
  df.isnull().sum()
  df.shape
  df.info()
  df.describe()

Step 3:  Ensure column names do not contain spaces (necessary for database storage and processing).
# Rename columns to remove spaces
  df.columns = df.columns.str.replace(" ", "_")
  df.columns
Next handle missing values to avoid issues in model training:
# Check for missing values
  print(df.isnull().sum())
# Encode categorical columns
  encoder = LabelEncoder()
  df['Role'] = encoder.fit_transform(df['Role'])
  df['Companies'] = encoder.fit_transform(df['Companies'])
  df['Location'] = encoder.fit_transform(df['Location'])
  df['Skills'] = encoder.fit_transform(df['Skills'])
# Identify non-numeric values in numerical columns
  print(df[['Min_Salary', 'Max_Salary', 'Experience']].applymap(lambda x: isinstance(x, str))) 
# Convert numerical columns to numeric, replacing errors with NaN
  df['Min_Salary'] = pd.to_numeric(df['Min_Salary'], errors='coerce')
  df['Max_Salary'] = pd.to_numeric(df['Max_Salary'], errors='coerce')
  df['Experience'] = pd.to_numeric(df['Experience'], errors='coerce')  
# Fill missing values with median
  df['Min_Salary'].fillna(df['Min_Salary'].median(), inplace=True)
  df['Max_Salary'].fillna(df['Max_Salary'].median(), inplace=True)
  df['Experience'].fillna(df['Experience'].median(), inplace=True)
# Scale numerical columns to standardize the range
  scaler = StandardScaler()
  df[['Min_Salary', 'Max_Salary', 'Experience']] = scaler.fit_transform(df[['Min_Salary', 'Max_Salary', 'Experience']])
  print("Data preprocessing completed!")


Step 4: Define input features (Experience, Company Size, Skills) and target variables(Salary):
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import joblib  
# Define features and target variable
  X = df[['Experience', 'Company_Size', 'Skills']]
  y = df['Min_Salary']

Step 5: train a Linear Regression model for salary prediction and use K-Means for clustering job roles:
# Split data into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  
# Check for missing values before training
  print(X_train.isnull().sum())
# Drop columns where all values are NaN (if any exist)
  X_train = X_train.dropna(axis=1, how='all')  
  X_test = X_test.dropna(axis=1, how='all')  
# Apply imputation for missing values in numerical columns
  numerical_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
  imputer = SimpleImputer(strategy="median")
  X_train[numerical_cols] = imputer.fit_transform(X_train[numerical_cols])
  X_test[numerical_cols] = imputer.transform(X_test[numerical_cols])
  print("Missing values handled successfully!")
# Train Linear Regression Model
  salary_model = LinearRegression()
  salary_model.fit(X_train, y_train)
  print("Salary Prediction Model Trained!")
#Train K-Means Clustering Model
  kmeans = KMeans(n_clusters=3, random_state=42)
  df['Cluster'] = kmeans.fit_predict(X)
  print("K-Means Clustering Model Trained!")
#Save trained models
  joblib.dump(salary_model, 'salary_model.pkl')
  joblib.dump(kmeans, 'kmeans_model.pkl')
  print("Models saved successfully!")

Step 6: connect to PostgreSQL on Railway.app and create a table to store salary predictions:
import psycopg2
# PostgreSQL connection URL from Railway.app looks like: "postgresql://your_username:your_password@your_host:your_port/your_database"
DB_URL = "postgresql://postgres:IfYbzdzkSkSWKJaxuPsryDzpYFLeKlQR@roundhouse.proxy.rlwy.net:39423/railway"
try:
    # Connect to PostgreSQL
    conn = psycopg2.connect(DB_URL)
    print("Successfully connected to Railway PostgreSQL!")
    # Create a cursor object
    cur = conn.cursor()
    # Create a table if it doesn’t exist
    cur.execute("""
        CREATE TABLE IF NOT EXISTS salary_predictions (
            id SERIAL PRIMARY KEY,
            role VARCHAR(255),
            company VARCHAR(255),
            location VARCHAR(255),
            experience FLOAT,
            company_size VARCHAR(50),
            skills TEXT,
            predicted_salary FLOAT
        );
    """)
    # Commit changes and close the connection
    conn.commit()
    cur.close()
    conn.close()
    print("Table created successfully!")
except Exception as e:
    print(f"Error connecting to PostgreSQL: {e}")

Step 7: Now, lets create a function to save and store predictions in PostgreSQL:
def save_prediction_to_db(role, company, location, experience, company_size, skills, predicted_salary):
    try:
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()
        # Insert prediction into the database
        cur.execute("""
            INSERT INTO salary_predictions (role, company, location, experience, company_size, skills, predicted_salary)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (role, company, location, experience, company_size, skills, predicted_salary))
        conn.commit()
        cur.close()
        conn.close()
        print("Prediction saved to database!")
    except Exception as e:
        print(f"Error saving prediction: {e}")

Step 8: To verify saved predictions, we fetch them from PostgreSQL.
# Retrieve saved predictions
conn = psycopg2.connect(DB_URL)
cur = conn.cursor()
cur.execute("SELECT * FROM salary_predictions;")
rows = cur.fetchall()
cur.close()
conn.close()
# Display saved predictions in Jupyter Notebook
for row in rows:
    print(row)

Step 9: Now, lets test salary predictions by calling our API:
import requests
# API Endpoint (Make sure your API is running)
url = "https://your-railway-app-url/predict"
# Provide job details for salary prediction
data = {
    "role": "Software Engineer",
    "company": "Amazon",
    "location": "Hyderabad",
    "experience": 3,
    "company_size": "5000+",
    "skills": "Python, SQL, AWS",
}
# Send the request
headers = {"Content-Type": "application/json"}
response = requests.post(url, json=data, headers=headers)
# Display the response
print("Response Status Code:", response.status_code)
print("Predicted Salary Response:", response.json())








