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
from models.text2brain_model import Text2BrainModel


application = Flask(__name__)
application.config.from_object(Config)
application.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(application)

q = Queue(connection=conn)

MODEL = None
OUTPUTS_DIR = "outputs"


class Result(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.String())
    saved_path = db.Column(db.String())
    hash_value = db.Column(db.String())

    def __repr__(self):
        return '<id {}>'.format(self.id)


def to_img(model, text):
    """ Output brain image """
    text = text.replace("/", "")
    hash_value = hashlib.new('sha512', text.encode()).hexdigest()
    saved_img_path = os.path.join(OUTPUTS_DIR, f'{hash_value}.nii.gz')

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
    # info['title'] = title
    # info['title_fontsize'] = title_fontsize
    return(info)

def save_visualizer_to_html(img, hash_value):
    view = plotting.view_img_on_surf(img, surf_mesh='fsaverage', symmetric_cmap=False, threshold='90%')
    view.save_as_html(f'{hash_value}.lh.html')


@application.route('/start', methods=['POST'])
def start_process():
    from flask_server import to_img
    data = json.loads(request.data.decode())
    print('Input string:', data['url_key'])

    job = q.enqueue_call(func=to_img,
                         args=(MODEL, data['url_key']),
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
    fc_channels = 64
    decoder_filters = 32

    checkpoint_file = f"checkpoints/fc{fc_channels}_d{decoder_filters}_relu_lr0.03_decay1e-06_drop0.55_seed28_checkpoint.pth"
    pretrained_bert_dir = "scibert_scivocab_uncased"

    """Init Model"""
    MODEL = Text2BrainModel(
        out_channels=1,
        fc_channels=fc_channels,
        decoder_filters=decoder_filters,
        pretrained_bert_dir=pretrained_bert_dir,
        drop_p=0.55)

    device = torch.device('cpu')
    state_dict = torch.load(checkpoint_file, map_location=device)['state_dict']
    MODEL.load_state_dict(state_dict)
    MODEL.eval()
    MODEL.to(device)

    application.debug = True
    application.run()
