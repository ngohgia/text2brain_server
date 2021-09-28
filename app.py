import os
import json
import hashlib
from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.dialects.postgresql import JSON
from config import Config

from rq import Queue
from rq.job import Job
from worker import conn

import numpy as np
import torch
import nibabel as nib
from nilearn import plotting
from nilearn import surface
from nilearn._utils.niimg_conversions import check_niimg_3d
from nilearn.plotting.html_surface import full_brain_info
from models.text2brain_model import Text2BrainModel, init_pretrained_model
from lookup import PaperIndex


application = Flask(__name__)
application.config.from_object(Config)
application.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(application)

q = Queue(connection=conn)

from results import *  # don't import just Result or a circular dependencies error will happen
from comments import *

model = init_pretrained_model(application.config['CHECKPOINT_PATH'], application.config['SCIBERT_DIR'], fc_channels=64, decoder_filters=32)
librarian = PaperIndex(application.config['TRAIN_IMAGES_DIR'], application.config['TRAIN_CSV'])

def to_img(text):
    """ Output brain image """
    hash_value = hashlib.new('sha512', text.encode()).hexdigest()
    saved_img_path = os.path.join(application.config['OUTPUTS_DIR'], f'{hash_value}.nii.gz')

    try:
        with torch.no_grad():
            pred = model((text, )).cpu().numpy().squeeze(axis=(0, 1))

        vol_data = np.zeros((46, 55, 46))
        vol_data[3:-3, 3:-4, :-6] = pred

        affine = np.array([[   4.,    0.,    0.,  -90.],
           [   0.,    4.,    0., -126.],
           [   0.,    0.,    4.,  -72.],
           [   0.,    0.,    0.,    1.]])

        pred_img = nib.Nifti1Image(vol_data, affine)
        nib.save(pred_img, saved_img_path)
    except Exception as err:
        return {"error": str(err)}

    try:
        result = Result(text=str(text), saved_path=saved_img_path, hash_value=hash_value, count=1)
        db.session.add(result)
        db.session.commit()
        return result.id
    except Exception as err:
        return {"error": "Unable to add item to database."}

def get_surface_object(stat_map_img):
    stat_map_img = check_niimg_3d(stat_map_img)
    info = full_brain_info(
        volume_img=stat_map_img, mesh='fsaverage', threshold="80%",
        cmap=plotting.cm.cold_hot, black_bg=False, vmax=None, vmin=None,
        symmetric_cmap=True, vol_to_surf_kwargs={})
    info['colorbar'] = True
    info['cbar_height'] = 0.5
    info['cbar_fontsize'] = 25
    return(info)

def render_result(result_id):
    data = Result.query.filter_by(id=result_id).first()
    try:
      img = nib.load(data.saved_path)
      info = get_surface_object(img)
      vol = img.get_fdata()[3:-3, 3:-4, :-6] # offset to fit mask
      related_articles = librarian.query(vol)
      return jsonify({"surface_info": info, "related_articles": related_articles})
    except Exception as err:
      return {"error": "Unable to retrieve result."}

def init_db():
    examples = application.config['EXAMPLES']
    for item in examples:
      text, hash_value = item
      saved_img_path = os.path.join(application.config['OUTPUTS_DIR'], f'{hash_value}.nii.gz')

      try:
        result = Result(text=text, saved_path=saved_img_path, hash_value=hash_value, count=1)
        db.session.add(result)
        db.session.commit()
      except Exception as err:
        print("Error initializing DB", err)


@application.route('/start', methods=['POST'])
def start_process():
    from app import to_img
    data = json.loads(request.data.decode())
    text = data['query'].lower().replace("/", "")[:application.config['MAX_QUERY_LENGTH']]

    job = q.enqueue_call(func=to_img,
                         args=(text, ),
                         result_ttl=5000)
    return job.get_id()

@application.route('/check', methods=['POST'])
def check():
    data = json.loads(request.data.decode())
    text = data['query'].lower().replace("/", "")[:application.config['MAX_QUERY_LENGTH']]

    existing_results = Result.query.filter_by(text=text)
    if existing_results.count() > 0:
      result = existing_results.first()
      result.count = result.count + 1
      db.session.commit()
      return jsonify({"id": result.id})
    else:
      return jsonify({"id": -1})

@application.route('/feel-brainy', methods=['POST'])
def feelBrainy():
    # assume that DB has been initialized with init_db
    rand_id = np.random.randint(1, len(application.config["EXAMPLES"])+1)

    existing_results = Result.query.filter_by(id=rand_id)
    if existing_results.count() > 0:
      result = existing_results.first()
      result.count = result.count + 1
      db.session.commit()
      return jsonify({"id": result.id, "query": result.text})
    else:
      return jsonify({"error": "No examples found"})

@application.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@application.route("/results/<result_id>", methods=['GET'])
def get_result(result_id):
    return render_result(result_id)

@application.route("/job_results/<job_key>", methods=['GET'])
def get_result_from_job(job_key):
    job = Job.fetch(job_key, connection=conn)

    if job.is_finished :
        return render_result(job.result)
    else:
        return "Processing...", 202

@application.route('/create-comment', methods=['POST'])
def createComment():
    data = json.loads(request.data.decode())
    try:
      comment = Comment(text=data['query'], comment=data["comment"], createdAt=1000) # createdAt=int(round(time.time())))
      db.session.add(comment)
      db.session.commit()
      return {"success": "true"}
    except Exception as err:
        return {"error": "Unable to add comment to database."}

if __name__ == '__main__':
    application.debug = True
    application.run("0.0.0.0")
