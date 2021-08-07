import os
basedir = os.path.abspath(os.path.dirname(__file__))


class Config(object):
    DEBUG = False
    TESTING = False
    CSRF_ENABLED = True
    # SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', 'postgresql:///giving_prof_dev')
    SQLALCHEMY_DATABASE_URI = f'sqlite:///{basedir}/results.db'
