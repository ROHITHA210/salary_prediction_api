# Use Python 3.9
FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy all files from your repo to the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Flask runs on
EXPOSE 5000

# Start the Flask app using Gunicorn
#CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080"]

