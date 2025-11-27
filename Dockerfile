# Versi Ringan untuk Railway
FROM tensorflow/serving:latest

# Copy model saja (Tanpa config monitoring)
COPY ./serving_model_dir /models/churn-model

# Set nama model
ENV MODEL_NAME=churn-model

# Expose port (Penting buat Railway)
EXPOSE 8501
