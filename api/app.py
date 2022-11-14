# FROM ubuntu:latest
FROM python:3.10.4
COPY ./*.py /exp/
COPY ./requirements.txt /exp/requirements.txt
COPY ./api/app.py /exp/api/app.py
COPY ./svm_gamma=0.001_C=0.7.joblib /exp/svm_gamma=0.001_C=0.7.joblib
RUN pip3 install -U scikit-learn
RUN pip3 install --no-cache-dir -r /exp/requirements.txt
WORKDIR /exp
EXPOSE 5000
# CMD ["python3", "./plot_graph.py"]
CMD ["python3","./api/app.py"]
# for build: docker build -t exp:v1 -f docker/Dockerfile .
# for running a container: docker run -p 5000:5000 -it exp:v1


"""from flask import Flask
from flask import request
from joblib import load

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/sum",methods=['POST'])
def sum():
    print(request.json)
    x=request.json['x']
    y=request.json['y']
    z=x+y
    return jsonify({'sum':z})"""

#model_path="svm_gamma=0.0005_C=2.joblib"
#@app.route("/predict",methods=['POST'])
#def predict_digit():
#    image= request.json['image']
#    model=load(model_path)
#    print("done_loading")
#    predicted=model.predict([image])

#    return {"y_predicted":predicted}
