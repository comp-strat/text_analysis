#!/usr/bin/env python
# -*- coding: UTF-8

# Preprocessing webpages for document embeddings and topic modeling
# Project title: Charter school identities 
# Creator: Jaren Haber, PhD Candidate
# Institution: Department of Sociology, University of California, Berkeley
# Date created: April 4, 2019
# Date last edited: April 4, 2019


## Import libraries and modules

import pandas as pd # for working with dataframes
import numpy as np # for working with numbers
import pickle # For working with .pkl files
import datetime # For working with dates & times
import sys # For terminal tricks
import re # For parsing text
from tqdm import tqdm # Shows progress over iterations, including in pandas via "progress_apply"

# For text parsing & modeling
#from sklearn.feature_extraction import text
from nltk import word_tokenize, sent_tokenize # widely used text tokenizer
from gensim.models.phrases import Phrases # Makes word2vec more robust: Looks not just at  To look for multi-word phrases within word2vec
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer() # approximate but effective (and common) method of stemming words

# For fast, more accurate text tokenization
import spacy
nlp = spacy.load('en', disable=['ner']) #,parser,tagger
nlp.remove_pipe('ner') #nlp.remove_pipe('parser') #nlp.remove_pipe('tagger')


# Import functions
sys.path.insert(0, '../../data_tools/')
from clean_text import stopwords_make, punctstr_make, unicode_make, clean_sentence
from quickpickle import quickpickle_dump, quickpickle_load # For quickly loading & saving pickle files in Python
from df_tools import check_df, load_filtered_df # For quick DF stats

# Define stopwords, unicode, punctstr
stop_words_list = stopwords_make()
unicode_list = unicode_make()
punctstr = punctstr_make()

def master_string_make(tupslist):
    """Extract text into master text string for each school.
    Cleans and tokenizes sentences, removing punctuation and numbers and making words lower-case.
    Loops over four nested levels, which from high to low are: tuple, chunk, sentence, word.
    
    Args:
        list of four-element tuples, the last element of which holds the long string of text we care about
    Returns:
        Master string for each school/website"""
    

    len_site = len(tupslist) # Count number of pages
    known_pages = set() # Initialize list of known pages for a school
    school_string = '' # Initialize master string for text of all a school's pages
            
    # Iterate over pages
    if len_site == 0 or not tupslist: # If site is empty, continue to next site without appending
        return
                
    for pagenum in range(len_site):
        sents_combined = ''
        if (tupslist[pagenum][3] in known_pages) or (tupslist[pagenum][3]==''): 
            continue # Skip this page if exactly the same as a previous page on this school's website
                
        for chunk in tupslist[pagenum][3].split("\n"): # Iterate over text chunks
            for sent in sent_tokenize(chunk): # sent_tokenize(chunk): # Iterate over sentences
                if ((sent == []) or (len(sent) == 0) or sent=="" or not sent): # If sentence is empty, continue to next sentence without appending
                    continue
                        
                # Filter out emails and URLs, remove punctuation:
                sent = " ".join(
                    [ps.stem(re.sub(r"["+punctstr+"]+", r'', word).strip("'").strip("-").strip(" ")) 
                    for word in sent.split() if 
                    word and 
                    "@" not in word and not 
                    word.startswith(('http', 'https', 'www', '//', '\\', 'x_', 'x/', 'srcimage')) and not
                    word.endswith(('.com', '.net', '.gov', '.org', '.jpg', '.pdf', 'png', 'jpeg', 'php'))])
                    
                sents_combined += ('\n' + sent) # Add sentence to list of sentences

        known_pages.add(tupslist[pagenum][3]) # Add page to known page list
        school_string += ('\n' + sents_combined) # Add to master string 
                            
    if school_string != '' and school_string not in ["", "\n", 0, "0"] and len(school_string)>0 and school_string != None:
        return(school_string)
    
    
## Load & prep data
data = load_filtered_df('../../nowdata/charters_2015.pkl', ['WEBTEXT'])
len_original = len(webtext)
data = data[data["WEBTEXT"] != ''][data["WEBTEXT"].notna()] # Drop where WEBTEXT is empty

tqdm.pandas(desc="Making clean master strings")
webtext = [] # Initialize list of master strings
webtext = data['WEBTEXT'].progress_apply(master_string_make)
webtext = webtext.dropna()

print("# rows in cleaned data before dropping empty WEBTEXT: ", str(len_original))
print("# rows in cleaned data after dropping empty WEBTEXT: ", len(webtext))

# Save data to disk
quickpickle_dump(webtext, "../data/webtext_quickcleaned.pickle")


## Detect and tag phrases
print("Detecting and tagging phrases in website text...")

# Threshold represents a threshold for forming the phrases (higher means fewer phrases). 
# A phrase of words a and b is accepted if (cnt(a, b) - min_count) * N / (cnt(a) * cnt(b)) > threshold, 
# where N is the total vocabulary size. By default this value is 10.0.

# Detect phrases in sentences based on collocation counts
phrases = Phrases(sentences=[site for site in webtext], 
                  delimiter=b'_', common_terms=stop_words_list, 
                  threshold=10, min_count=10) 

# Apply phrase detection model to each sentence in data, while removing digits
webtext = [sent_tokenize(site) for site in list(webtext)]
webtext = " ".join(
                   [phrases[
                            [word for word in sentence if not word.isdigit()]
                           ].strip() 
                   for sentence in tqdm(
                                        sent_tokenize(webtext), desc="Parsing phrases"
                                       )
                   ]
                  )

# Save data to disk
quickpickle_dump(webtext, "../data/webtext_quickcleaned_phrased.pickle")

sys.exit() # Kill script when done, just to be safe