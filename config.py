import os
basedir = os.path.abspath(os.path.dirname(__file__))

DB_DIR = os.getenv('TEXT2BRAIN_DB_DIR')

class Config(object):
    DEBUG = False
    TESTING = False
    CSRF_ENABLED = True
    SQLALCHEMY_DATABASE_URI = f'sqlite:///{DB_DIR}/results.db'
    CHECKPOINT_PATH = os.getenv('TEXT2BRAIN_CHECKPOINT')
    SCIBERT_DIR = os.getenv('TEXT2BRAIN_SCIBERT_DIR')
    OUTPUTS_DIR = os.getenv('TEXT2BRAIN_OUTPUTS_DIR')
