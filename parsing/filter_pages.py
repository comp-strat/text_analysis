
# coding: utf-8

# In[2]:


#necessary imports 
import csv
import pandas as pd
import os
import re
import string
import collections


import numpy as np
from itertools import groupby
from sklearn.feature_extraction import text
import nltk
from nltk.corpus import stopwords
import datetime
import logging


import string
#folder_prefix = '/home/jovyan/work/'


# In[3]:

folder_prefix = '/vol_b/data/'

logging.basicConfig(filename="filter_pages250.log", level=logging.INFO)


original_df = pd.read_pickle(folder_prefix + 'nowdata/charters_2015.pkl')


# In[6]:


#original_df


# In[20]:


#original_df['WEBTEXT'][0]


# In[7]:


keywords = ['values', 'academics', 'academic', 'skills', 'skill', 'purpose', 'purposes',
                       'direction', 'mission', 'vision', 'visions', 'missions',
                       'ideals', 'cause', 'causes', 'curriculum', 'curricular',
                       'method', 'methods', 'pedagogy', 'pedagogical', 'pedagogies', 'approach', 'approaches', 'model', 'models', 'system', 'systems',
                       'structure', 'structures', 'philosophy', 'philosophical', 'philosophies', 'beliefs', 'believe', 'belief',
                       'principles', 'principle', 'creed', 'creeds', 'credo', 'moral', 'morals', 'morality', 'history', 'histories', 'our story',
                       'the story', 'school story', 'background', 'backgrounds', 'founding', 'founded', 'foundation', 'foundations', 'foundational',
                       'established','establishment', 'our school began', 'we began',
                       'doors opened', 'school opened', 'about us', 'our school', 'who we are',
                       'identity', 'identities', 'profile', 'highlights']


# In[8]:


keywords_values = {'values':2, 'academics':1, 'academic':1, 'skills':1, 'skill':1, 'purpose':2, 'purposes':2,
                       'direction':1, 'mission':2, 'vision':2, 'visions':2, 'missions':2,
                       'ideals':2, 'cause':1, 'causes':1, 'curriculum':2, 'curricular':2,
                       'method':1, 'methods':1, 'pedagogy':2, 'pedagogical':1, 'pedagogies':1, 'approach':1, 'approaches':1, 'model':2, 'models':2, 'system':2, 'systems':2,
                       'structure':1, 'structures':1, 'philosophy':2, 'philosophical':2, 'philosophies':2, 'beliefs':2, 'believe':2, 'belief':2,
                       'principles':2, 'principle':2, 'creed':2, 'creeds':2, 'credo':2, 'moral':2, 'morals':2, 'morality':2, 'history':1, 'histories':1, 'our story':1,
                       'the story':1, 'school story':1, 'background':1, 'backgrounds':1, 'founding':1, 'founded':1, 'foundation':1, 'foundations':1, 'foundational':1,
                       'established':1,'establishment':1, 'our school began':1, 'we began':1,
                       'doors opened':1, 'school opened':1, 'about us':2, 'our school':1, 'who we are':1,
                       'identity':1, 'identities':1, 'profile':1, 'highlights':2}


# In[9]:


# Hybrid approach

# Separate keywords to be treated differently
small_keywords = []
large_keywords = []

for entry in keywords:
    small_keywords.append(entry) if len(entry.split()) < 3 else large_keywords.append(entry)

large_words = [entry.split() for entry in large_keywords] # list words for each large dict entry
large_lengths = [len(x) for x in large_words]
large_first_words = [x[0] for x in large_words] # first words of each large entry in dict


# In[10]:


def dict_count2(text):

    """Hybrid of dict_count and dict_count1. 
    
    Uses dict_count1 approach to count matches for entries with > 2 words in keywords.
    Uses dict_count approach for all other entries.
    """

    counts = 0 # hitscore
    splitted_phrase = re.split('\W+|_', text.lower()) # Remove punctuation with regex that keeps only letters and spaces

    for length in range(1, 3):
        if len(splitted_phrase) < length:
            continue # If text chunk is shorter than length of dict entries being matched, there are no matches.
        for i in range(len(splitted_phrase) - length + 1):
            entry = ' '.join(splitted_phrase[i:i+length]) # Builds chunk of 'length' words without ending space
            if entry in keywords:
                counts += keywords_values[entry]
    mask = [[word == entry for word in splitted_phrase] for entry in large_first_words]
    indices = np.transpose(np.nonzero(mask))
    for ind in indices:
        if ind[1] <= (len(splitted_phrase) - large_lengths[ind[0]]) and large_words[ind[0]] == splitted_phrase[ind[1] : ind[1] + large_lengths[ind[0]]]:
            counts += keywords_values[large_keywords[ind[0]]]
    return counts


# In[11]:


# def getKey(item):
#     return item[0]

def takeSecond(elem):
    return elem[1]



# In[21]:

num = 0

def filter_pages(li_tuples, MIN_HITCOUNT, MAX_NUMPAGES):
    """
    Takes in a list of quadruples
    string texts from a school. Most likely from the WEBTEXT column
    For the row, the function returns the top 250 pages of the list who have the highest hitcount    
    """
    global num
    
    #just keep track of the hit count
    
    #turn the list of quadruples into a list of just string texts  
    
    if len(li_tuples) == 0: #if taking in an emoty list, we return an empty list
        return li_tuples
    
    school_pages = []
    
    for tup in li_tuples:
        if len(tup) == 4:
            school_pages.append(tup[3])
    
    
    li_pairs = []
    #index = 0
    for page in school_pages:
        hit_count = dict_count2(page.lower())
        if hit_count >= MIN_HITCOUNT:
            li_pairs.append([page, hit_count])
        
        #index+=1
    
    #sort the tuples/sublists by highest to lowest hit count
    #take the top 250 , or less is len(school_pages) < 250
    sorted_tuples = sorted(li_pairs, key=takeSecond, reverse = True)
    filtered_tuples = []
    if len(sorted_tuples) < MAX_NUMPAGES:
        filtered_tuples = sorted_tuples
    else:
        filtered_tuples = sorted_tuples[:MAX_NUMPAGES]
    
    #get the page at the correpsonding index (tup[1]) from school_pages
    final_pages = [tup[0] for tup in filtered_tuples] 
    if final_pages is not None :
        logging.info("row done : " + str(num))
    else:
        logging.info("row is None : " + str(num))
        final_pages = []
    num = num + 1
    
    return final_pages
    
    
            
        
        
    


# In[22]:


# def filter_pages2(school_pages, MIN_HITCOUNT = 1, MAX_NUMPAGES = 250, AGGRO = False, is_set = False):
#     """Filters page text with hit count at least min hit count if school has more than MAX_NUMPAGES distinct pages else unfiltered of pages is returned.
    
#     Returns max_numpages pages with priority given to higher hitscore and then lower page depth(even when AGGRO is TRUE). Boolean value returned is to help generate WEBTEXT_METHOD later.
#     school_pages: entry of 'webtext' column
#     is_set: True if school_pages is set of pages
#     aggro: When true, only pages that have >= MIN_HITCOUNT hits pass. Only resort to CMO pages when no pages pass
#     """
#     if not is_set:
#         school_pages = set([p for p in school_pages])
# #     if len(pages) <= MAX_NUMPAGES:
# #         return ([(p.url, p.boo, p.depth, p.text) for p in pages], 0)
#     all_tuples = []
#     filtered_num = 0 # number of pages that passed the hitscore requirement
#     filtered = []
#     max_hc = -1
#     min_depth = 99999
#     for p in school_pages:
#         hit_count = dict_count2(p)
#         if hit_count >= MIN_HITCOUNT:
#             filtered.append(hit_count, p)
#             filtered_num += 1
# #         if max_hc < hit_count:
# #             max_hc = hit_count
# #         if min_depth > int(p.depth):
# #             min_depth = int(p.depth)
#         # maintain list containing all pages and corresponding hit scores
#         all_tuples.append((hit_count, (p.url, p.boo, p.depth, p.text)))
#     if not aggro and filtered_num and filtered_num <= MAX_NUMPAGES:
#             return ([t[1] for t in filtered], False)        
#     all_tuples = [(t[0] - .00001*int(t[1][2]), t[1]) for t in all_tuples] # prepare list to be heapified
#     if aggro:
#         all_tuples = filtered
#     # priority number is hit_count - .00001*page.depth so pages with high hitscores are prioritized followed by low page depths
#     filtered = [t[1] for t in q.nlargest(MAX_NUMPAGES, all_tuples)]
#     return (filtered, filtered_num == False)



# In[23]:

def middle(original_df):
    index = 0    
    webtext= original_df['WEBTEXT'].apply(lambda row: filter_pages(row, MIN_HITCOUNT = 1, MAX_NUMPAGES = 250))
    return webtext

#     global num
#     num = 0

#new_webtext = middle(original_df)
#new_webtext = original_df['WEBTEXT'].apply(filter_pages)
original_df['WEBTEXT'] = original_df['WEBTEXT'].apply(lambda row_li: filter_pages(row_li, 1, 250))
filtered_df = original_df[['NCESSCH', 'WEBTEXT']]
#filtered_df['WEBTEXT'] = new_webtext

filtered_df.to_pickle(folder_prefix + "nowdata/parsing/filtered_250.pkl")
#print(num)