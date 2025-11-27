FROM tensorflow/serving:latest

# --- PERBAIKAN PATH (SESUAI GAMBAR GITHUB) ---
# Kita copy ISI folder timestamp (1764208728) ke folder versi "1" di dalam container
COPY ./serving_model_dir/1764208728 /models/churn-model/1

ENV MODEL_NAME=churn-model

# Expose port untuk API
EXPOSE 8501
