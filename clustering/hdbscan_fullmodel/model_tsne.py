
# coding: utf-8

# In[8]:


import pickle


# In[4]:


from sklearn.decomposition import PCA


# In[1]:


from sklearn.manifold import TSNE
import numpy as np
import gensim

wem_newpath = "../Charter-school-identities/data/wem_model_train250_nostem_unlapped_300d.txt"
model = gensim.models.KeyedVectors.load_word2vec_format(wem_newpath)


# In[2]:


word_vecs = model[model.vocab]
row_sums = np.linalg.norm(word_vecs, axis=1)
unit_vecs = word_vecs / row_sums[:, np.newaxis]


# In[9]:


with open ('fullmodel_labels', 'rb') as fp:
    full_labels = pickle.load(fp)


# In[28]:


words = np.array(list(model.vocab))[np.where(full_labels!=-1)[0]]


# In[30]:


label_vecs = unit_vecs[np.where(full_labels!=-1)[0]]


# In[33]:


pca = PCA(n_components = 50)
pca.fit(label_vecs)
transformed = pca.transform(label_vecs)


# In[ ]:


projection = TSNE(perplexity=50.0).fit_transform(label_vecs)
np.save('TSNE_approx',projection)

