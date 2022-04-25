from pathlib import Path
import numpy as np
import pandas as pd


class PaperIndex:
  def __init__(self, haystack_file, train_csv):
    self.corpus = pd.read_csv(train_csv, usecols=['title', 'pmid', 'author'], dtype=str).sort_values('pmid')
    self.haystack = np.load(haystack_file)
    self.ssB = (self.haystack**2).sum(1)

  def demean(self, masked_vol):
    assert len(masked_vol) == self.haystack.shape[-1]
    out = masked_vol.astype(np.float16)
    return out - out.mean()

  def corr_coeff(self, A_mA):
    # Sum of squares across rows
    ssA = (A_mA**2).sum(axis=1, keepdims=True)

    # corr coeff
    num = np.dot(A_mA, self.haystack.T)
    den = np.sqrt(np.dot(ssA, self.ssB[None]))
    return np.where(den, num / den, 0)  # return 0 correlation when denominator is 0

  def query(self, masked_vol, topk=5):
    needle = np.expand_dims(self.demean(masked_vol), axis=0)
    corr = self.corr_coeff(needle).squeeze()
    indices = np.argsort(corr)[-topk:][::-1]
    out = self.corpus.iloc[indices].to_dict('records')
    for i, entry in enumerate(out):
        entry['correlation'] = float(corr[indices[i]])
    # return list of dict with 4 keys 'pmid', 'title', 'author', and 'correlation'
    return out