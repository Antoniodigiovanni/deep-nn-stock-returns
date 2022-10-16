#FROM google/cloud-sdk:latest
#RUN gsutil cp gs://thesis-data-1/data.zip /data

FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

WORKDIR /app

RUN apt-get update \
  && apt-get install -y wget \
  && apt-get install -y unzip \
  && rm -rf /var/lib/apt/lists/*

RUN wget -O data.zip --no-check-certificate "https://onedrive.live.com/download?cid=9053A48EF4F6502C&resid=9053A48EF4F6502C%21129&authkey=ACwFbiwWDeuqnm4"
RUN unzip data.zip
RUN rm data.zip

#COPY /data .

RUN pip3 install pandas
RUN pip3 install statsmodels
RUN pip3 install nni
RUN pip3 install captum
RUN pip3 install wrds
RUN pip3 install seaborn
RUN pip3 install tensorboard
RUN mkdir saved
RUN cd saved
RUN mkdir results
RUN cd /app

COPY ./src ./src

#RUN touch ./saved/file_created_in_docker.txt
# RUN nnictl create --config ./src/tuning/experiment_config.yaml --port 8080
# CMD ["sleep", "infinity"]
# CMD ["nnictl","create","--config","./src/tuning/experiment_config.yaml","--port","8080","&&","sleep","infinity"]
CMD ["python","./src/main.py","--saveDirName","TestDocker","--expandingTuning"]
