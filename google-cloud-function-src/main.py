import os
from google.cloud import storage
import base64
import hashlib
import functions_framework

from text2brain_model import init_pretrained_model
from lookup import PaperIndex
import torch
import numpy as np

# TO BE REPLACED BY YOUR SETTINGS
MODEL_BUCKET_NAME  = "google_cloud_bucket"
PROJECT_ID         = "google_cloud_project_id"
T2B_MODEL_FILE     = "text2brain_checkpoint.pth"
SCIBERT_DIR        = "scibert_scivocab_uncased"
SCIBERT_CONFIG     = "config.json"
SCIBERT_VOCAB      = "vocab.txt"
SCIBERT_MODEL      = "pytorch_model.bin"
MASK_FILE          = "mask.npy"

TRAIN_IMG_ARR      = "train_img_by_pmid.npy"
TRAIN_CSV          = "train.csv"

LOCAL_DIR          = "/tmp"

LOCAL_MODEL_FILE   = os.path.join(LOCAL_DIR, T2B_MODEL_FILE)
LOCAL_SCIBERT_DIR  = os.path.join(LOCAL_DIR, SCIBERT_DIR)
LOCAL_TRAIN_IMG_FILE = os.path.join(LOCAL_DIR, TRAIN_IMG_ARR)
LOCAL_TRAIN_CSV    = os.path.join(LOCAL_DIR, TRAIN_CSV)

model = None
librarian = None

storage_client   = storage.Client(PROJECT_ID)
model_bucket = storage_client.get_bucket(MODEL_BUCKET_NAME)

# initialization
if not os.path.exists(LOCAL_DIR):
  os.makedirs(LOCAL_DIR)
mask_blob  = model_bucket.blob(MASK_FILE)
mask_blob.download_to_filename(os.path.join(LOCAL_DIR, MASK_FILE))
mask = np.load(os.path.join(LOCAL_DIR, MASK_FILE))

def download_model_files(bucket):
  # Download text2brain model
  t2b_blob     = bucket.blob(T2B_MODEL_FILE)
  t2b_blob.download_to_filename(LOCAL_MODEL_FILE)  

  # Download SciBERT assets
  if not os.path.exists(LOCAL_SCIBERT_DIR):
    os.makedirs(LOCAL_SCIBERT_DIR)
  scibert_config_blob  = bucket.blob(SCIBERT_CONFIG)
  scibert_config_blob.download_to_filename(os.path.join(LOCAL_SCIBERT_DIR, SCIBERT_CONFIG))

  scibert_vocab_blob  = bucket.blob(SCIBERT_VOCAB)
  scibert_vocab_blob.download_to_filename(os.path.join(LOCAL_SCIBERT_DIR, SCIBERT_VOCAB))

  scibert_model_blob  = bucket.blob(SCIBERT_MODEL)
  scibert_model_blob.download_to_filename(os.path.join(LOCAL_SCIBERT_DIR, SCIBERT_MODEL))

def download_librarian_files(bucket):
  train_img_arr_blob  = bucket.blob(TRAIN_IMG_ARR)
  train_img_arr_blob.download_to_filename(LOCAL_TRAIN_IMG_FILE)

  train_csv_blob      = bucket.blob(TRAIN_CSV)
  train_csv_blob.download_to_filename(LOCAL_TRAIN_CSV)

def predict(request):
  '''
  request is a JSON with two keys:
  - query: input text for generating a new brain image
  - img_name: an existing image name to be retrieved from past prediction.
    If img_name is empty, a new prediction is made from query
  '''

  if request.method == 'OPTIONS':
    # Allows GET requests from any origin with the Content-Type
    # header and caches preflight response for an 3600s
    headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Max-Age': '3600'
    }

    return ('', 204, headers)

  # Set CORS headers for the main request
  headers = {
      'Access-Control-Allow-Origin': '*'
  }

  global model, librarian

  params = request.get_json()

  if not librarian:
    download_librarian_files(model_bucket)
    librarian = PaperIndex(LOCAL_TRAIN_IMG_FILE, LOCAL_TRAIN_CSV)

  # make a new prediction
  if not model:
    download_model_files(model_bucket)
    model = init_pretrained_model(LOCAL_MODEL_FILE, LOCAL_SCIBERT_DIR)

  try:
    text = params["query"]
    if len(text) == 0:
      return({ "init": "done" }, 200, headers) # return a dummy response

    with torch.no_grad():
      pred = model((text, )).cpu().numpy().squeeze(axis=(0, 1))[mask > 0]

      hash_value = hashlib.new('sha512', text.encode()).hexdigest()

      related_articles = librarian.query(pred)
      return ({ "result": pred.tolist(), "img_name": hash_value, "related_articles": related_articles }, 200, headers)
  except Exception as error:
      return { "error": str(error) }
