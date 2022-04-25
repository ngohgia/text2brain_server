import os
import json
import hashlib
import requests
import random
from flask import Flask, render_template, request, redirect, jsonify, url_for, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.dialects.postgresql import JSON
from config import Config

import numpy as np
import logging
import time
import nibabel as nib
from nilearn import plotting
from nilearn import surface
from nilearn._utils.niimg_conversions import check_niimg_3d
from nilearn.plotting.html_surface import full_brain_info

application = Flask(__name__)
application.config.from_object(Config)
application.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(application)

from results import *  # don't import just Result or a circular dependencies error will happen
from comments import *

mask  = np.load(application.config['MASK_FILE'])

LOCAL_OUTPUTS_DIR = application.config["OUTPUTS_DIR"]

def _get_surface_object(stat_map_img):
    stat_map_img = check_niimg_3d(stat_map_img)
    info = full_brain_info(
        volume_img=stat_map_img, mesh='fsaverage5', threshold="80%",
        cmap=plotting.cm.cold_hot, black_bg=False, vmax=None, vmin=None,
        symmetric_cmap=True, vol_to_surf_kwargs={})
    info['colorbar'] = True
    info['cbar_height'] = 0.5
    info['cbar_fontsize'] = 25
    return(info)

def _get_random_example_query():
    rand_id = np.random.randint(len(application.config["EXAMPLES"]))
    return application.config["EXAMPLES"][rand_id]


@application.route('/feel-brainy', methods=['POST'])
def feel_brainy():
    query = _get_random_example_query()
    return _predict_or_retrieve(query)

@application.route('/', methods=['GET'])
def index():
    query = request.args.get('q', None)
    if query is None:
      query = _get_random_example_query()
      return redirect(url_for('index', q=query))
    return render_template('index.html')

@application.route('/create-comment', methods=['POST'])
def create_comment():
    data = json.loads(request.data.decode())
    try:
      comment = Comment(text=data['query'], comment=data["comment"], createdAt=int(round(time.time())))
      db.session.add(comment)
      db.session.commit()
      return {"success": "true"}
    except Exception as err:
      return {"error": "Unable to add comment to database: %s" % str(err)}

@application.route('/download/<string:filename>', methods=['GET'])
def download(filename):
    return send_from_directory(LOCAL_OUTPUTS_DIR, filename, as_attachment=True)

@application.route('/predict', methods=['POST'])
def predict():
    data = json.loads(request.data.decode())

    return _predict_or_retrieve(data["query"])

@application.route('/api', methods=['POST'])
def api():
    data = json.loads(request.data.decode())

    return _predict_or_retrieve(data["query"], for_api=True)

@application.route('/init', methods=['POST'])
def init_model():
    r = requests.post(application.config["GCF_URL"], json={ "query": ""}).json()
    return r


def _save_brain_img(img_name, pred, mask):
    vol_data = np.zeros((46, 55, 46))
    affine = np.array([[   4.,    0.,    0.,  -90.],
       [   0.,    4.,    0., -126.],
       [   0.,    0.,    4.,  -72.],
       [   0.,    0.,    0.,    1.]])

    cropped_vol = np.zeros((40, 48, 40))
    cropped_vol[mask] = pred
    vol_data[3:-3, 3:-4, :-6] = cropped_vol
    pred_img = nib.Nifti1Image(vol_data, affine)

    local_img_path = os.path.join(LOCAL_OUTPUTS_DIR, "%s.nii.gz" % img_name)
    nib.save(pred_img, local_img_path)

    return pred_img

def _predict_or_retrieve(raw_query, for_api=False):
    query = raw_query.lower().strip().replace("/", "")[:application.config['MAX_QUERY_LENGTH']]

    try:
      existing_results = Result.query.filter_by(text=query)
      if existing_results.count() > 0:
        result = existing_results.first()

        result.count = result.count + 1
        db.session.commit()

        img_name = result.img_name
        related_articles = json.loads(result.related_articles)
        local_img_path = os.path.join(LOCAL_OUTPUTS_DIR, "%s.nii.gz" % img_name)
        pred_img = nib.load(local_img_path)
      else:
        r = requests.post(application.config["GCF_URL"], json={ "query": query}).json()
        img_name = r["img_name"]
        related_articles = r["related_articles"]

        result = Result(text=query, img_name=img_name, related_articles=json.dumps(related_articles), count=1)
        db.session.add(result)
        db.session.commit()

        pred = np.asarray(r["result"])
        pred_img = _save_brain_img(img_name, pred, mask)

      res ={
        "query": query,
        "download_path": url_for("download", filename="%s.nii.gz" % img_name, _external=True),
        "related_articles": related_articles
      }

      if not for_api:
        info = _get_surface_object(pred_img)
        res["surface_info"] = info
    
      return jsonify(res)
    except Exception as err:
      return {"error": "Unable to retrieve result: %s" % str(err)}

if __name__ == '__main__':
    application.debug = True
    application.run("0.0.0.0")
else:
    gunicorn_logger = logging.getLogger('gunicorn.error')
    application.logger.handlers = gunicorn_logger.handlers
    application.logger.setLevel(gunicorn_logger.level)
