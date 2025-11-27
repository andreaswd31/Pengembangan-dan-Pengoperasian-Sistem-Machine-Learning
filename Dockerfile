FROM tensorflow/serving:latest

# --- PERBAIKAN PATH ---
# Kita copy ISI dari folder timestamp spesifik ke folder versi "1" di dalam container
# GANTI '1764035919' dengan angka timestamp yang ada di folder serving_model_dir Anda!
COPY ./serving_model_dir/andreaswd31-dicoding_pipeline_project2/1764035919 /models/churn-model/1

ENV MODEL_NAME=churn-model

# Penting untuk Railway
EXPOSE 8501
