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


application = Flask(__name__)
application.config.from_object(Config)
application.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(application)

q = Queue(connection=conn)

from results import *  # don't import just Result or a circular dependencies error will happen

model = init_pretrained_model(application.config['CHECKPOINT_PATH'], application.config['SCIBERT_DIR'], fc_channels=64, decoder_filters=32)

def to_img(text):
    """ Output brain image """
    text = text.replace("/", "")
    hash_value = hashlib.new('sha512', text.encode()).hexdigest()
    saved_img_path = os.path.join(application.config['OUTPUTS_DIR'], f'{hash_value}.nii.gz')

    try:
        with torch.no_grad():
            print(model)
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
        print(err)
        return {"error": str(err)}
    print('Saving to:', saved_img_path)

    try:
        result = Result(text=text, saved_path=saved_img_path, hash_value=hash_value)
        db.session.add(result)
        db.session.commit()
        return result.id
    except Exception as err:
        print(err)
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


@application.route('/start', methods=['POST'])
def start_process():
    from flask_server import to_img
    data = json.loads(request.data.decode())
    print('Input string:', data['url_key'])

    job = q.enqueue_call(func=to_img,
                         args=(data['url_key'], ),
                         result_ttl=5000)
    return job.get_id()

@application.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@application.route("/results/<job_key>", methods=['GET'])
def get_results(job_key):
    job = Job.fetch(job_key, connection=conn)

    if job.is_finished:
        data = Result.query.filter_by(id=job.result).first()
        try:
            img = nib.load(data.saved_path)
            info = get_surface_object(img)
            return jsonify(info)
        except Exception as err:
            return {"error": "Unable to retrieve result."}
    else:
        return "Processing...", 202

if __name__ == '__main__':
    application.debug = True
    application.run("0.0.0.0")
