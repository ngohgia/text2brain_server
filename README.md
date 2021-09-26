# text2brain
Generating brain activation maps from free-form text query

- Create conda environment from env.yml
- Download checkpoints from [Google Drive](https://drive.google.com/file/d/13Gc0M4i4zj16aVtZzGUQs4oPcoyRTGWw/view?usp=sharing) and do `tar -xzvf checkpoints.tar.gz`
- Download uncased SciBert pretrained model from [AllenAI S3](https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/huggingface_pytorch/scibert_scivocab_uncased.tar) and unzip into `scibert_scivocab_uncased` directory
- Run `python predict_cpu_only.py <input_query> <output_file>`, e.g `python predict_cpu_only.py "self-generated thought" prediction.nii.gz`

## MAC

brew install libressl
brew install openssl

Install pyenv
```
CFLAGS="-I$(brew --prefix openssl)/include -I$(brew --prefix bzip2)/include -I$(brew --prefix readline)/include -I$(xcrun --show-sdk-path)/usr/include" LDFLAGS="-L$(brew --prefix openssl)/lib -L$(brew --prefix readline)/lib -L$(brew --prefix zlib)/lib -L$(brew --prefix bzip2)/lib" pyenv install --patch 3.6.13 < <(curl -sSL https://github.com/python/cpython/commit/8ea6353.patch\?full_index\=1)
```
Install redis

pip install --upgrade pip # 21.2.4

from flask_server import db
