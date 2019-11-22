# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.7-stretch

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# Install production dependencies.

RUN apt-get update -y
RUN apt-get install -y python-pip python-dev build-essential

RUN pip install -r requirements.txt

# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.

#CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 app:app
ENTRYPOINT ["python"]
CMD ["app.py"]