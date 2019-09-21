
# coding: utf-8

# In[ ]:


import imp
from sklearn.manifold import TSNE
import numpy as np
import gensim

wem_newpath = "../Charter-school-identities/data/wem_model_train250_nostem_unlapped_300d.txt"
model = gensim.models.KeyedVectors.load_word2vec_format(wem_newpath)


# In[ ]:


word_vecs = model[model.vocab]
row_sums = np.linalg.norm(word_vecs, axis=1)
unit_vecs = word_vecs / row_sums[:, np.newaxis]
projection = TSNE().fit_transform(unit_vecs)
np.save('TSNE_full',projection)

