FROM python:3.10
COPY . /app
WORKDIR /app
RUN apt-get update && apt-get install -y git git-lfs
# RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh
# RUN apt-get install git-lfs
RUN git lfs install
RUN pip install -r requirements.txt
EXPOSE 8000
CMD uvicorn main:app --reload --host 0.0.0.0 --port 8000