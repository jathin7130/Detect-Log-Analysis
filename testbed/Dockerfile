FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir flask requests
COPY common /app/common
ARG SERVICE
COPY ${SERVICE}/app.py /app/app.py
CMD ["python", "app.py"]
