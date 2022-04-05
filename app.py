import os
import json
import hashlib
import requests
import random
from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.dialects.postgresql import JSON
from config import Config

import numpy as np
import time
import nibabel as nib
from nilearn import plotting
from nilearn import surface
from nilearn._utils.niimg_conversions import check_niimg_3d
from nilearn.plotting.html_surface import full_brain_info
from lookup import PaperIndex


application = Flask(__name__)
application.config.from_object(Config)
application.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(application)

from results import *  # don't import just Result or a circular dependencies error will happen
from comments import *

librarian = PaperIndex(application.config['TRAIN_IMG_BY_PMID_FILE'], application.config['TRAIN_CSV'])
mask  = np.load(application.config['MASK_FILE'])

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


@application.route('/feel-brainy', methods=['POST'])
def feel_brainy():
    rand_id = np.random.randint(len(application.config["EXAMPLES"]))

    query = application.config["EXAMPLES"][rand_id]
    return _predict_or_retrieve(query)

@application.route('/', methods=['GET'])
def index():
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

@application.route('/predict', methods=['POST'])
def predict():
    data = json.loads(request.data.decode())

    query = data["query"].lower().strip().replace("/", "")[:application.config['MAX_QUERY_LENGTH']]
    return _predict_or_retrieve(query)

def _predict_or_retrieve(query):
    try:
      existing_results = Result.query.filter_by(text=query)
      if existing_results.count() > 0:
        result = existing_results.first()
        result.count = result.count + 1
        db.session.commit()

        r = requests.post(application.config["GCF_URL"], json={ "query": query, "img_name": result.img_name })
        if "result" not in r.json():
          r = requests.post(application.config["GCF_URL"], json={ "query": query})
      else:
        r = requests.post(application.config["GCF_URL"], json={ "query": query})
        img_name = r.json()["img_name"]

        result = Result(text=query, img_name=img_name, count=1)
        db.session.add(result)
        db.session.commit()

      pred = np.asarray(r.json()["result"])

      vol_data = np.zeros((46, 55, 46))
      affine = np.array([[   4.,    0.,    0.,  -90.],
         [   0.,    4.,    0., -126.],
         [   0.,    0.,    4.,  -72.],
         [   0.,    0.,    0.,    1.]])

      cropped_vol = np.zeros((40, 48, 40))
      cropped_vol[mask] = pred
      vol_data[3:-3, 3:-4, :-6] = cropped_vol
      pred_img = nib.Nifti1Image(vol_data, affine)

      info = get_surface_object(pred_img)

      application.logger.info(pred.shape)
      related_articles = librarian.query(pred)

      return jsonify({"query": query, "surface_info": info, "related_articles": related_articles})
    except Exception as err:
      return {"error": "Unable to retrieve result: %s" % str(err)}


if __name__ == '__main__':
    application.debug = True
    application.run("0.0.0.0")
