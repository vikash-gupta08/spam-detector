# Use official Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy your Python script to the container
COPY spamDectorModel.py .

# Install required Python packages
RUN pip install pandas scikit-learn streamlit requests

# Expose Streamlit's default port
EXPOSE 8501

# Run Streamlit app on container start
CMD ["streamlit", "run", "spamDectorModel.py"]