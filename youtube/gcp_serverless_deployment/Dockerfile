FROM python:3.9.5

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

RUN pip install torch==1.13.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu

COPY . .

EXPOSE 5000

ENTRYPOINT ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]