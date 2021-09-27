from pathlib import Path
import numpy as np
import pandas as pd


class PaperIndex:
    def __init__(self, image_folder):
        root = Path(image_folder)
        self.mask = np.load(root/'mask.npy')

        self.corpus = pd.read_csv('train.csv', usecols=['title', 'pmid', 'author'], dtype=str).sort_values('pmid')
        self.haystack = np.zeros((len(self.corpus), self.mask.sum()), dtype=np.float16)
        for i, pmid in enumerate(self.corpus['pmid']):
            fp = root/f'pmid_{pmid}.npy'
            if not fp.exists():
                raise Exception(f'Image of PMID {pmid} missing')
            self.haystack[i] = self.row_of_voxels(np.load(fp))
        self.ssB = (self.haystack**2).sum(1)

        print('Initialized')

    def row_of_voxels(self, image):
        assert image.shape == self.mask.shape, f'unexpected image shape {image.shape}'
        out = image[self.mask].astype(np.float16)
        return out - out.mean()

    def corr_coeff(self, A_mA):
        # Sum of squares across rows
        ssA = (A_mA**2).sum(axis=1, keepdims=True)

        # corr coeff
        num = np.dot(A_mA, self.haystack.T)
        den = np.sqrt(np.dot(ssA, self.ssB[None]))
        return np.where(den, num / den, 0)  # return 0 correlation when denominator is 0

    def query(self, image, topk=5):
        needle = self.row_of_voxels(image).reshape(1, -1)
        corr = self.corr_coeff(needle).squeeze()
        indices = np.argsort(corr)[-topk:][::-1]
        out = self.corpus.iloc[indices].to_dict('records')
        for i, entry in enumerate(out):
            entry['correlation'] = corr[indices[i]]
        # return list of dict with 4 keys 'pmid', 'title', 'author', and, 'correlation'
        return out



if __name__ == '__main__':
    librarian = PaperIndex('data/processed/images/')
    out = librarian.query(np.load('data/processed/images/pmid_10523407.npy'))
    print(out)
    assert out[0]['pmid'] == '10523407'
    out = librarian.query(np.load('data/processed/images/pmid_10666559.npy'))
    print(out)
    assert out[0]['pmid'] == '10666559'
