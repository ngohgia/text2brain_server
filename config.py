import os
basedir = os.path.abspath(os.path.dirname(__file__))

DB_DIR = os.getenv('TEXT2BRAIN_DB_DIR')

class Config(object):
    MAX_QUERY_LENGTH = 140
    DEBUG = False
    TESTING = False
    CSRF_ENABLED = True
    SQLALCHEMY_DATABASE_URI = f'sqlite:///{DB_DIR}/text2brain.db'
    TRAIN_IMG_BY_PMID_FILE = os.getenv('TEXT2BRAIN_TRAIN_IMAGE_BY_PMID_FILE')
    TRAIN_CSV = os.getenv('TEXT2BRAIN_TRAIN_CSV')
    MASK_FILE = os.getenv('TEXT2BRAIN_MASK_IMG')
    OUTPUTS_DIR = os.getenv('TEXT2BRAIN_OUTPUTS_DIR')
    
    GCF_URL = os.getenv('TEXT2BRAIN_GCF_URL')

    EXAMPLES = [
      "self-generated thought",
      "viewing faces",
      "feeling happy",
      "thinking out loud",
      "working memory",
      "imagine all the people sharing all the world",
      "oh i believe in yesterday",
      "reading a book",
      "listening to music",
    ]
