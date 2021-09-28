import os
basedir = os.path.abspath(os.path.dirname(__file__))

DB_DIR = os.getenv('TEXT2BRAIN_DB_DIR')

class Config(object):
    MAX_QUERY_LENGTH = 140
    DEBUG = False
    TESTING = False
    CSRF_ENABLED = True
    SQLALCHEMY_DATABASE_URI = f'sqlite:///{DB_DIR}/text2brain.db'
    CHECKPOINT_PATH = os.getenv('TEXT2BRAIN_CHECKPOINT')
    SCIBERT_DIR = os.getenv('TEXT2BRAIN_SCIBERT_DIR')
    OUTPUTS_DIR = os.getenv('TEXT2BRAIN_OUTPUTS_DIR')
    TRAIN_IMAGES_DIR = os.getenv('TEXT2BRAIN_TRAIN_IMAGES_DIR')
    TRAIN_CSV = os.getenv('TEXT2BRAIN_TRAIN_CSV')

    EXAMPLES = [
      ["self-generated thought", "cdc02bf8130d3223b5fcdd8bfa60079daa2b6915c4722e14916f15940f64b22d55a6a484856394eab78aef38b9dad05da3974d194c8c7c8151fc66c3960e0fce"],
      ["viewing faces", "85558c59216ee4e5235ef606557b35b8c276d8c6c862a2b63b346fcc0d6e538b582c87710e1593d345ae1e5cf2a74faa7a8ca215f14bb820709d44baeaeaac9b"],
      ["feeling happy", "2774b96e54a068f7bee97dcda85fb649a4deb7006a6b2247ff9cb5a4b0350173f491375b89c9039395f1bcc274d68246336ec56b9ea6a0c6dfa880948990e27c"],
      ["thinking out loud", "80f9ae5dfbc24d27929e58b0e68a0d627257072c5563e33cab96c6cf085b4253a551de3726696f6995fcdaf38e159f94130b09f8471d600602855b077c878a56"],
      ["working memory", "fa416898314d36b50dc66ec735b90782ec76c99d3e7b97df98027e203ad75f1bfa7c7b0f7e22b449d95789b7fca40e0025f983b285d387c19136509e2c213319"],
      ["imagine all the people sharing all the world", "b9b12231ce1c8e0fe6367d9dc74bf11598393e9072bb93622a8aa0f0bc7e2f8fbec4665c8290f99dd2665771dec0499117381f264c3c2dbd9e4e6b8790503cde"],
      ["oh i believe in yesterday", "96417de864da0a2e9a1949805ff5c22767cbc1acb3f7fc35b7cda1eb1567f2688453d681b6a822b5530334e0151e2b60b1f3e4507b6552ac2f54fd357b49d3f3"]
    ]
