FROM python:3.11-slim

WORKDIR /app 

COPY . .

RUN pip install fastapi uvicorn pandas onnxruntime

EXPOSE 8000

CMD [ "uvicorn","api:app","--host","0.0.0.0" , "--port", "8000" ]