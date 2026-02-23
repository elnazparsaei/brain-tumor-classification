FROM tensorflow/tensorflow:2.16.1-gpu

WORKDIR /app

RUN pip install --no-cache-dir --ignore-installed blinker

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY models/best_model.keras ./models/best_model.keras

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]

    