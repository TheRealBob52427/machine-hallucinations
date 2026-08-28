# Use an official lightweight Python image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend and frontend directories into the container
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Expose the port your app runs on (adjust if your app uses a different port)
EXPOSE 8000

# Set the command to run your backend application
CMD ["python", "backend/main.py"]
