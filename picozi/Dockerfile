#Use python as base image
FROM python:3.8.5-slim

#Use working directory /app
WORKDIR /app

#copy all the content of current directory to /app
ADD . /app

#installing required packages from pypi
RUN pip3 install --trusted-host pypi.python.org -r requirements.txt

#open the port 5000
EXPOSE 5000

#set environment variable
ENV NAME OpentoAll

#run python program
CMD ["python", "app.py"]
