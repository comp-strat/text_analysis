import pandas as pd
import time
import re
import numpy as np
import ast
import sys


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
charter_path = '../../charters_full_2015.pkl'
df_charter = pd.read_pickle(charter_path)
df_charter['WEBTEXT']=df_charter['WEBTEXT'].fillna('') # turn nan to empty list/string for future convenience
df_charter['CMO_WEBTEXT'] = df_charter['CMO_WEBTEXT'].fillna('0') # ugly hack so that we can apply literal_eval on column later
df_charter['CMO_WEBTEXT'] = df_charter['CMO_WEBTEXT'].apply(ast.literal_eval) # apply to whole column
df_charter['CMO_WEBTEXT'] = df_charter['CMO_WEBTEXT'].replace(0, '') # now all nan are '' in both WEBTEXT columns

# Optimized dict_count attempt for cases where entries in 'custom_dict' have long word lengths

# precalculations
dict_words = [entry.split() for entry in keywords] # list words for each dict entry
dict_lengths = [len(x) for x in dict_words]
first_words = [x[0] for x in dict_words] # first words of each entry in dict

def dict_count1(text):
    words_list = re.split('\W+', text) # list of words in text
    # find indices where word in first_words matches word in words_list
    mask = [[word == entry for word in words_list] for entry in first_words]
    indices = np.transpose(np.nonzero(mask))
    count = 0
    for ind in indices:
        if ind[1] <= (len(words_list) - dict_lengths[ind[0]]) and dict_words[ind[0]] == words_list[ind[1] : ind[1] + dict_lengths[ind[0]]]:
            count+=1
    return count

# Repurposed Jaren Haber's dict_count and helper function in webparser_mp.py. Bug fixed on chunk building.
max_entry_length = max([len(entry.split()) for entry in keywords]) # Get length (in words) of longest entry in combined dictionary

def dict_count(text):

    """Performs dictionary analysis, returning number of dictionary hits found.
    Removes punctuation and stems the phrase being analyzed.
    Compatible with multiple-word dictionary elements."""

    counts = 0 # number of matches between text_list and custom_dict
    splitted_phrase = re.split('\W+', text) # Remove punctuation with regex that keeps only letters and spaces

    # Do dictionary analysis for word chunks of lengths max_entry_length down to 1
    for length in range(1, max_entry_length + 1):
        if len(splitted_phrase) < length:
            continue # If text chunk is shorter than length of dict entries being matched, there are no matches.
        for i in range(len(splitted_phrase) - length + 1):
            entry = ' '.join(splitted_phrase[i:i+length]) # Builds chunk of 'length' words without ending space
            if entry in keywords:
                counts += 1

    return counts

# hybrid approach

# separate keywords to be treated differently
small_keywords = []
large_keywords = []

for entry in keywords:
    small_keywords.append(entry) if len(entry.split()) < 3 else large_keywords.append(entry)

large_words = [entry.split() for entry in large_keywords] # list words for each large dict entry
large_lengths = [len(x) for x in large_words]
large_first_words = [x[0] for x in large_words] # first words of each large entry in dict

def dict_count2(text):

    """Hybrid of dict_count and dict_count1. Uses dict_count1 approach to count matches for entries with > 2 words in keywords.
    Uses dict_count approach for all other entries.
    """

    counts = 0 # number of matches between text_list and custom_dict
    splitted_phrase = re.split('\W+', text) # Remove punctuation with regex that keeps only letters and spaces

    # Do dictionary analysis for word chunks of lengths max_entry_length down to 1
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

def filter_pages(school_pages, MIN_HITCOUNT = 1):
    """Returns the list of page text with hit count at least min hit count.

    Also filters out duplicate text.
    school_pages: entry of 'webtext' column
    """
    pages = set([Page(p) for p in school_pages])
    all_tuples = []
    filtered = []
    max_hc = -1
    min_depth = 99999
    for p in pages:
        hit_count = dict_count2(p.text)
        if hit_count >= MIN_HITCOUNT:
            filtered.append((p.url, p.boo, p.depth, p.text))
        if max_hc < hit_count:
            max_hc = hit_count
        if min_depth > int(p.depth):
            min_depth = int(p.depth)
        all_tuples.append(((p.url, p.boo, p.depth, p.text),hit_count))
    if  filtered:
        return (filtered, False)
    else:
        if max_hc == 0:
            return ([t[0] for t in all_tuples if int(t[0][2]) == min_depth], True)
        return ([t[0] for t in all_tuples if t[1] == max_hc], True)
    # return [page for page in set(school_pages) if dict_count2(page[3])>=MIN_HITCOUNT] # maintains tuples but does not handle case where tuple is different but text is same
def run_filter(type, MIN_HITCOUNT = 1.0):
    """Runs filter of given type. Creates checkpoint file with column of filtered pages. Column name is of form 'CMO_FILTERED_TEXT#' for type 'c' and 'FILTERED_TEXT#' for type 'w' where # is min hit count.

    type: column to run filter on. 'c' for CMO_WEBTEXT, 'w' for 'WEBTEXT'
    MIN_HITCOUNT: min hit count to pass filter
    """
    if type == 'w':
        print('WEBTEXT Page filter start. Min hit count: {:f}'.format(MIN_HITCOUNT))
        filtered_pages = []
        s = []
        start = time.time()
        for i, row in enumerate(df_charter['WEBTEXT'].values):
            result = filter_pages(row, MIN_HITCOUNT)
            filtered_pages.append(result[0])
            s.append(result[1])
            if i%1000 == 0:
                end = time.time()
                print('Time Elapsed:{:f}, Percent Complete:{:f}'.format(end - start,i*100/len(df_charter)))
        df_charter['WEBTEXT'] = pd.Series(filtered_pages, index=df_charter.index)
        df_charter['WEBTEXT_EMPTY'] = pd.Series(s, index=df_charter.index)
        ckpt_file_path = 'charters_full_2015{:s}{:d}_checkpoint1.pkl'.format(type,round(10*MIN_HITCOUNT))
        df_charter.to_pickle(ckpt_file_path) # checkpoint file contains new column 'FILTERED_TEXT'
        print('Completed text filtering. Saved checkpoint to ' + ckpt_file_path)
    elif type == 'c':
        print('CMO_WEBTEXT Page filter start. Min hit count: {:f}'.format(MIN_HITCOUNT))
        filtered_pages = []
        s = []
        start = time.time()
        for i, row in enumerate(df_charter['CMO_WEBTEXT'].values):
            result = filter_pages(row, MIN_HITCOUNT)
            filtered_pages.append(result[0])
            s.append(result[1])
            if i%1000 == 0:
                end = time.time()
                print('Time Elapsed:{:f}, Percent Complete:{:f}'.format(end - start,i*100/len(df_charter)))
        df_charter['CMO_WEBTEXT'] = pd.Series(filtered_pages, index=df_charter.index)
        df_charter['CMO_WEBTEXT_EMPTY'] = pd.Series(s, index=df_charter.index)
        ckpt_file_path = 'charters_full_2015{:s}{:d}_checkpoint1.pkl'.format(type,round(10*MIN_HITCOUNT))
        df_charter.to_pickle(ckpt_file_path) # checkpoint file contains new column 'FILTERED_TEXT'
        print('Completed text filtering. Saved checkpoint to ' + ckpt_file_path)
    elif type == 'a':
        print('Complete Page filter start. Min hit score: {:f}'.format(MIN_HITCOUNT))
        filtered_pages = []
        s = []
        start = time.time()
        for i, row in enumerate(df_charter['WEBTEXT'].values):
            result = filter_pages(row, MIN_HITCOUNT)
            if result[1]:
                result_cmo = filter_pages(df_charter.loc[df_charter.index[i], 'CMO_WEBTEXT'],MIN_HITCOUNT)
                if result_cmo[1]:
                    filtered_pages.append(result[0])
                    s.append(2)
                else:
                    filtered_pages.append(result_cmo[0])
                    s.append(1)
            else:
                filtered_pages.append(result[0])
                s.append(0)
            if i%1000 == 0:
                end = time.time()
                print('Time Elapsed:{:f}, Percent Complete:{:f}'.format(end - start,i*100/len(df_charter)))
        df_charter['WEBTEXT'] = pd.Series(filtered_pages, index=df_charter.index)
        df_charter['WEBTEXT_METHOD'] = pd.Series(s, index=df_charter.index) # 2 empty webtext, empty cmo_webtext, 1 empty_webtext, non empty cmo_webtext, 0 nonempty webtext
        ckpt_file_path = 'charters_full_2015{:s}{:d}_checkpoint1.pkl'.format(type,round(MIN_HITCOUNT*10))
        df_charter.to_pickle(ckpt_file_path) # checkpoint file contains modified 'WEBTEXT' column
        print('Completed text filtering. Saved checkpoint to ' + ckpt_file_path)

class Page:
    def __init__(self,p):
        self.url = p[0]
        self.boo = p[1]
        self.depth = p[2]
        self.text = p[3]
    def __repr__(self):
        return self.text
    def __eq__(self, other):
        if isinstance(other, Page):
            return self.text == other.text
        else:
            return False
    def __ne__(self, other):
        return (not self.__eq__(other))
    def __hash__(self):
        return hash(self.__repr__())

# only checks filter type not min hit score. min hit score must be covertible to float
if sys.argv[1] == 'c':
    run_filter('c', float(sys.argv[2]))
elif sys.argv[1] == 'w':
    run_filter('w', float(sys.argv[2]))
elif sys.argv[1] == 'a':
    run_filter('a', float(sys.argv[2]))
    df_charter['REPLACED'] = df_charter['WEBTEXT_METHOD'] == 1 # replaced wtih CMO filtered pages
    df_right = df_charter.groupby('CMO_NAME')['REPLACED'].sum() > 0 # df to be merged to the right of df_charter
    df_right = df_right.reset_index()
    df_right.rename(columns={"REPLACED": "CMO_REPLACED"},inplace=True)
    df_charter = pd.merge(df_charter, df_right, how = 'left', on = ['CMO_NAME'])
    ckpt_file_path = 'charters_full_2015_{:d}.pkl'.format(round(float(sys.argv[2])*10))
    df_charter.to_pickle(ckpt_file_path) # checkpoint file contains new 'CMO_REPLACED','WEBTEXT_METHOD', and filtered 'WEBTEXT' columns
    print('Completed text filtering. Saved checkpoint to ' + ckpt_file_path)
else:
    print('Invalid type. Use c, w, or a for cmo_webtext, webtext, and complete filtering, respectively.')
