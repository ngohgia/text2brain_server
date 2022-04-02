from pathlib import Path
import numpy as np
import pandas as pd


class PaperIndex:
    def __init__(self, haystack_file, train_csv):
        self.corpus = pd.read_csv(train_csv, usecols=['title', 'pmid', 'author'], dtype=str).sort_values('pmid')
        self.haystack = np.load(haystack_file)
        self.ssB = (self.haystack**2).sum(1)

        print('Initialized librarian')

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
        # return list of dict with 4 keys 'pmid', 'title', 'author', and, 'correlation'
        return out



if __name__ == '__main__':
    librarian = PaperIndex('data/train_img_by_pmid.npy', '/Users/ngohgia/Work/text2brain_server/data/processed/train.csv')
    mask = np.load("data/processed/images/mask.npy")
    raw_sample = np.load('data/processed/images/pmid_10523407.npy')
    masked_vol = raw_sample[mask > 0]
    out = librarian.query(masked_vol)
    print(out)
    assert out[0]['pmid'] == '10523407'
    raw_sample = np.load('data/processed/images/pmid_10666559.npy')
    masked_vol = raw_sample[mask > 0]
    out = librarian.query(masked_vol)
    print(out)
    assert out[0]['pmid'] == '10666559'
