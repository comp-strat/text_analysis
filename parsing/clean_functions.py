# Import packages
import re, datetime
import string # for one method of eliminating punctuation
from nltk.corpus import stopwords # for eliminating stop words


def stopwords_make():
    """Create stopwords list"""
                                                     
    stop_word_list = list(set(stopwords.words("english"))) # list of english stopwords

    # Add dates to stopwords
    for i in range(1,13):
        stop_word_list.append(datetime.date(2008, i, 1).strftime('%B'))
    for i in range(1,13):
        stop_word_list.append((datetime.date(2008, i, 1).strftime('%B')).lower())
    for i in range(1, 2100):
        stop_word_list.append(str(i))

    # Add other common stopwords
    stop_word_list.append('00') 
    stop_word_list.extend(['mr', 'mrs', 'sa', 'fax', 'email', 'phone', 'am', 'pm', 'org', 'com', 
                           'Menu', 'Contact Us', 'Facebook', 'Calendar', 'Lunch', 'Breakfast', 'FAQs', 'FAQ']) # web stopwords
    stop_word_list.extend(['el', 'en', 'la', 'los', 'para', 'las', 'san']) # Spanish stopwords
    stop_word_list.extend(['angeles', 'diego', 'harlem', 'bronx', 'austin', 'antonio']) # cities with many charter schools

    # Add state names & abbreviations (both uppercase and lowercase) to stopwords
    states = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL", 
              "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", 
              "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", 
              "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", 
              "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WI", "WV", "WY", 
              "Alabama", "Alaska", "Arizona", "Arkansas", "California", 
              "Colorado", "Connecticut", "District of Columbia", "Delaware", "Florida", 
              "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", 
              "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", 
              "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi", 
              "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", 
              "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota", 
              "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", 
              "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", 
              "Vermont", "Virginia", "Washington", "Wisconsin", "West Virginia", "Wyoming"]
    for state in states:
        stop_word_list.append(state)
    for state in [state.lower() for state in states]:
        stop_word_list.append(state)

    # Add to stopwords useless and hard-to-formalize words/chars from first chunk of previous model vocab (e.g., a3d0, \fs19)
    # First create whitelist of useful terms probably in that list, explicitly exclude from junk words list both these and words with underscores (common phrases)
    whitelist = ["Pre-K", "pre-k", "pre-K", "preK", "prek", 
                 "1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th", "11th", "12th", 
                 "1st-grade", "2nd-grade", "3rd-grade", "4th-grade", "5th-grade", "6th-grade", 
                 "7th-grade", "8th-grade", "9th-grade", "10th-grade", "11th-grade", "12th-grade", 
                 "1st-grader", "2nd-grader", "3rd-grader", "4th-grader", "5th-grader", "6th-grader", 
                 "7th-grader", "8th-grader", "9th-grader", "10th-grader", "11th-grader", "12th-grader", 
                 "1stgrade", "2ndgrade", "3rdgrade", "4thgrade", "5thgrade", "6thgrade", 
                 "7thgrade", "8thgrade", "9thgrade", "10thgrade", "11thgrade", "12thgrade", 
                 "1stgrader", "2ndgrader", "3rdgrader", "4thgrader", "5thgrader", "6thgrader", 
                 "7thgrader", "8thgrader", "9thgrader", "10thgrader", "11thgrader", "12thgrader"]
    with open(vocab_path_old) as f: # Load vocab from previous model
        junk_words = f.read().splitlines() 
    junk_words = [word for word in junk_words[:8511] if ((not "_" in word) 
                                                         and (not any(term in word for term in whitelist)))]
    stop_word_list.extend(junk_words)
                                                     
    return stop_word_list
                                                     
    
def punctstr_make():
    """Creates punctuations list"""
                    
    punctuations = list(string.punctuation) # assign list of common punctuation symbols
    #addpuncts = ['*','•','©','–','`','’','“','”','»','.','×','|','_','§','…','⎫'] # a few more punctuations also common in web text
    #punctuations += addpuncts # Expand punctuations list
    #punctuations = list(set(punctuations)) # Remove duplicates
    punctuations.remove('-') # Don't remove hyphens - dashes at beginning and end of words are handled separately)
    punctuations.remove("'") # Don't remove possessive apostrophes - those at beginning and end of words are handled separately
    punctstr = "".join([char for char in punctuations]) # Turn into string for regex later

    return punctstr
                                                     
                                                     
def unicode_make():
    """Create list of unicode chars"""
                    
    unicode_list  = []
    for i in range(1000,3000):
        unicode_list.append(chr(i))
    unicode_list.append("_cid:10") # Common in webtext junk
                                                     
    return unicode_list


def clean_sentence(sentence):
    """Removes numbers, emails, URLs, unicode characters, hex characters, and punctuation from a sentence 
    separated by whitespaces. Returns a tokenized, cleaned list of words from the sentence.
    
    Args: 
        Sentence, i.e. string that possibly includes spaces and punctuation
    Returns: 
        Cleaned & tokenized sentence, i.e. a list of cleaned, lower-case, one-word strings"""
    
    # Create useful lists using above functions:
    stop_words_list = stopwords_make()
    punctstr = puncstr_make()
    unicode_list = unicode_make()
    
    # Replace unicode spaces, tabs, and underscores with spaces, and remove whitespaces from start/end of sentence:
    sentence = sentence.replace(u"\xa0", u" ").replace(u"\\t", u" ").replace(u"_", u" ").strip(" ")
    
    # Remove hex characters (e.g., \xa0\, \x80):
    sentence = re.sub(r'[^\x00-\x7f]', r'', sentence) #replace anything that starts with a hex character 

    # Replace \\x, \\u, \\b, or anything that ends with \u2605
    sentence = re.sub(r"\\x.*|\\u.*|\\b.*|\u2605$", "", sentence)
        
    # Remove all elements that appear in unicode_list (looks like r'u1000|u10001|'):
    sentence = re.sub(r'|'.join(map(re.escape, unicode_list)), '', sentence)
    
    sentence = re.sub("\d+", "", sentence) # Remove numbers
    
    sent_list = [] # Initialize empty list to hold tokenized sentence (words added one at a time)
    
    for word in sentence.split(): # Split by spaces and iterate over words
        
        word = word.strip() # Remove leading and trailing spaces
        
        # Filter out emails and URLs:
        if ("@" not in word and not word.startswith(('http', 'https', 'www', '//', '\\', 'x_', 'x/', 'srcimage')) and not word.endswith(('.com', '.net', '.gov', '.org', '.jpg', '.pdf', 'png', 'jpeg', 'php'))):
            
            # Remove punctuation (only after URLs removed):
            word = re.sub(r"["+punctstr+"]+", r'', word).strip("'").strip("-") # Remove punctuations, and remove dashes and apostrophes only from start/end of words
            if word not in stop_word_list: # Filter out stop words
                
                sent_list.append(word.lower()) # Add lower-cased word to list

    return sent_list # Return clean, tokenized sentence