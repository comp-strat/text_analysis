#
# # coding: utf-8
#
# # # HDBSCAN clustering of Inquiry and Discipline dictionary wordvecs
#
# # ### Installing hdbscan
#
# # In[1]:
#
#
# get_ipython().system('pip install --upgrade numpy')
#
#
# # In[3]:
#
#
# get_ipython().system('pip install hdbscan')
#
#
# # In[1]:
#
#
# get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import hdbscan
import pandas as pd
import numpy as np
import gensim
import pickle

# define paths
wem_newpath = "../Charter-school-identities/data/wem_model_train250_nostem_unlapped_300d.txt"
dict_path = '/home/jovyan/work/Charter-school-identities/dicts/'


# In[2]:


# load model
model = gensim.models.KeyedVectors.load_word2vec_format(wem_newpath)

# load dicts/wordvecs
dict_list = []
word_vecs_list = []
core_list = []
core_word_list = []
word_list = []
dict_names = ['inquiry', 'discipline']
for name in dict_names:
    with open(dict_path+name+'.txt') as f:
        new_dict = f.read().splitlines()
        word_vecs = []
        core = []
        core_word = []
        word = []
        for i, entry in enumerate(new_dict):
            try:
                word_vecs.append(model[entry])
                word.append(entry)
                if i < 30:
                    core.append(model[entry])
                    core_word.append(entry)
            except:
                pass
        dict_list.append(new_dict)
        word_vecs_list.append(word_vecs)
        core_list.append(core)
        core_word_list.append(core_word)
        word_list.append(word)



flatui = ["#3498db", "#e74c3c","#9b59b6", "#34495e", "#2ecc71"] # custom colors


# In[6]:


def cluster_and_visualize(words, data, min_size = 5):
    """Clusters data using HDBSCAN and visualizes using TSNE

    min_size: min_cluster_size parameter for HDBSCAN
    data: list of wordvecs
    words: list of (string)words corresponding to data
    """
    clusterer = hdbscan.HDBSCAN(min_cluster_size = min_size, core_dist_n_jobs=-2)
    clusterer = clusterer.fit(data)
    # projection = TSNE().fit_transform(data)
    # color_palette = sns.color_palette(flatui)
    # cluster_colors = [color_palette[x] if x >= 0
    #                   else (0.5, 0.5, 0.5)
    #                   for x in clusterer.labels_]
    # cluster_member_colors = [sns.desaturate(x, p) for x, p in
    #                          zip(cluster_colors, clusterer.probabilities_)]
#     plt.scatter(*projection.T, s=50, linewidth=0, c=cluster_member_colors, alpha=.5)
    # plt.figure(figsize=(16, 16))
    # for i, coord in enumerate(projection):
    #     plt.scatter(*coord, s=50, linewidth=0, c=cluster_member_colors[i], alpha=.5)
    #     plt.annotate(words[i], (coord[0], coord[1]))
    return clusterer



# ## Full Model Clustering

# In[7]:


word_vecs = model[model.vocab]
row_sums = np.linalg.norm(word_vecs, axis=1)
unit_vecs = word_vecs / row_sums[:, np.newaxis]


clusterer = cluster_and_visualize(list(model.vocab),unit_vecs, 10)

with open('fullmodel_labels', 'wb') as fp:
    pickle.dump(clusterer.labels_, fp)

with open('fullmodel_probs', 'wb') as fp:
    pickle.dump(clusterer.probabilities_, fp)

# to load
# with open ('outfile', 'rb') as fp:
#     itemlist = pickle.load(fp)
