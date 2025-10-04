# Start with a standard Python base image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Create a non-root user for security
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Copy the requirements file and install dependencies
COPY --chown=user requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the rest of the application code
COPY --chown=user . .

# Expose the port the app runs on
EXPOSE 5000

# Command to run the application using Gunicorn (a production server)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]