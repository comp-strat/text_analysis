
# coding: utf-8

# In[7]:


import imp
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import hdbscan
import pandas as pd
import numpy as np
import gensim
import pickle

wem_newpath = "../Charter-school-identities/data/wem_model_train250_nostem_unlapped_300d.txt"


model = gensim.models.KeyedVectors.load_word2vec_format(wem_newpath)

word_vecs = model[model.vocab]
row_sums = np.linalg.norm(word_vecs, axis=1)
unit_vecs = word_vecs / row_sums[:, np.newaxis]

clusterer = hdbscan.HDBSCAN(min_cluster_size = 10, core_dist_n_jobs=-2)
clusterer = clusterer.fit(unit_vecs)

with open('fullmodel_labels', 'wb') as fp:
    pickle.dump(clusterer.labels_, fp)

with open('fullmodel_probs', 'wb') as fp:
    pickle.dump(clusterer.probabilities_, fp)

with open('fullmodel_persistence', 'wb') as fp:
    pickle.dump(clusterer.cluster_persistence_, fp)

