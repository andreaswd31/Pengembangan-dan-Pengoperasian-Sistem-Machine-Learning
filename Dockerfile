#  image resmi TensorFlow Serving
FROM tensorflow/serving:latest

# Copy model
COPY ./serving_model_dir /models/churn-model

COPY ./monitoring/prometheus.config /models/prometheus.config

# Set nama model
ENV MODEL_NAME=churn-model

CMD ["--monitoring_config_file=/models/prometheus.config"]

EXPOSE 8501
EXPOSE 8500