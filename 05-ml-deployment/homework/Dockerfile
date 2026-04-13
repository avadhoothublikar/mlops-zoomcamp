FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

WORKDIR /app

COPY homework.py homework.py

RUN pip install \
    numpy==1.21.6 \
    pandas \
    pyarrow \
    scikit-learn==1.0.2

ENTRYPOINT ["python", "homework.py"]