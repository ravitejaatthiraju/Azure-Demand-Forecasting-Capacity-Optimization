# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the local directory contents into the container at /app
# This includes api.py and the 'data' and 'models' folders
COPY . .

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define the command to run the app using Gunicorn
# This is a production-ready web server
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "api:app"]
