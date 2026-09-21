# Cloud Run image for the smart (GRU) model — DB role 4.
#
#   gcloud builds submit --tag gcr.io/PROJECT/f1-smart cloud/
#   gcloud run deploy f1-smart --image gcr.io/PROJECT/f1-smart \
#       --region asia-south1 --allow-unauthenticated --memory 2Gi
#
# Pick a region near you and note it in the report: the measured network
# latency is a property of the region you deployed to, not of "the cloud".
FROM python:3.11-slim

WORKDIR /app

# CPU-only torch — the GPU wheel is ~2GB and Cloud Run has no GPU.
# The image size directly affects cold-start time, which you are
# measuring, so this is not just tidiness.
RUN pip install --no-cache-dir \
      torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir \
      numpy pandas scikit-learn flask gunicorn

COPY models.py predict_service.py ./
COPY smart_model.pkl /opt/ml/smart_model.pkl
COPY cloud/lambda_handler.py ./

ENV SMART_MODEL_PATH=/opt/ml/smart_model.pkl
ENV PORT=8080
EXPOSE 8080

# Single worker, single thread: you are measuring per-request latency,
# and concurrency here would confound it with queueing inside the
# container. Scale with Cloud Run instances instead.
CMD ["gunicorn", "-b", "0.0.0.0:8080", "-w", "1", "--threads", "1", \
     "--timeout", "120", "lambda_handler:app"]
