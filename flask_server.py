import os
import json
import hashlib
from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from config import Config

from rq import Queue
from rq.job import Job
from worker import conn

import numpy as np
import torch
from models.text2brain_model import Text2BrainModel


application = Flask(__name__)
application.config.from_object(Config)
application.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(application)

q = Queue(connection=conn)

MODEL = None

def to_img(model, text):
    """ Output brain image """
    text = text.replace("/", "")
    hash_value = hashlib.new('sha512_256', text.encode()).hexdigest()
    saved_img_path = f'{hash_value}.nii.gz'

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
        result = Result(text=text, result=saved_img_path)
        db.session.add(result)
        db.session.commit()
        return result.id
    except:
        return {"error": "Unable to add item to database."}


@application.route('/start', methods=['POST'])
def start_process():
    from flask_server import to_img
    data = json.loads(request.data.decode())

    job = q.enqueue_call(func=to_img,
                         args=(MODEL, data['query']),
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
        return jsonify(data.result)  # path to saved image
    else:
        return "Nay!", 202


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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(checkpoint_file, map_location=device)['state_dict']
    MODEL.load_state_dict(state_dict)
    MODEL.eval()
    MODEL.to(device)

    application.debug = True
    application.run()
