#!/usr/bin/env python
# coding: utf-8

# In[47]:


import sqlite3
import csv
import pandas as pd
import os
import re

import math
from tqdm import tqdm
import multiprocessing as mp
import time
import sched


import logging
 
# add filemode="w" to overwrite
#logging.basicConfig(filename="df_chunk_index.log", level=logging.INFO)

#logging.basicConfig(filename="rows_overlaps_far.log", level=logging.INFO)


 



from difflib import SequenceMatcher as SeqMatcher
import numpy as np

# Import packages for multiprocessing
import os # For navigation


folder_prefix = '/vol_b/data/'


# In[48]:


new_data = pd.read_csv(folder_prefix + "nowdata/parsing/current_df4_WEBTEXT.csv", sep="\t", low_memory=False, encoding="utf-8")



# In[4]:


# keywords = ['values', 'academics', 'skills', 'purpose',
#                        'direction', 'mission', 'vision', 'vision', 'mission', 'our purpose',
#                        'our ideals', 'ideals', 'our cause', 'curriculum','curricular',
#                        'method', 'pedagogy', 'pedagogical', 'approach', 'model', 'system',
#                        'structure','philosophy', 'philosophical', 'beliefs', 'believe',
#                        'principles', 'creed', 'credo', 'values','moral', 'history', 'our story',
#                        'the story', 'school story', 'background', 'founding', 'founded',
#                        'established','establishment', 'our school began', 'we began',
#                        'doors opened', 'school opened', 'about us', 'our school', 'who we are',
#                        'our identity', 'profile', 'highlights']

# mission_keywords = ['mission','vision', 'vision:', 'mission:', 'our purpose', 'our ideals', 'ideals:', 'our cause', 'cause:', 'goals', 'objective']
# curriculum_keywords = ['curriculum', 'curricular', 'program', 'method', 'pedagogy', 'pedagogical', 'approach', 'model', 'system', 'structure']
# philosophy_keywords = ['philosophy', 'philosophical', 'beliefs', 'believe', 'principles', 'creed', 'credo', 'value',  'moral']
# history_keywords = ['history', 'story','our story', 'the story', 'school story', 'background', 'founding', 'founded', 'established', 'establishment', 'our school began', 'we began', 'doors opened', 'school opened']
# about_keywords =  ['about us', 'our school', 'who we are', 'overview', 'general information', 'our identity', 'profile', 'highlights']

# mission_keywords = set(stemmer.stem(word) for word in mission_keywords)
# curriculum_keywords = set(stemmer.stem(word) for word in curriculum_keywords)
# philosophy_keywords = set(stemmer.stem(word) for word in philosophy_keywords)
# history_keywords = set(stemmer.stem(word) for word in history_keywords)
# about_keywords =  set(stemmer.stem(word) for word in about_keywords)


# In[5]:


# m = '[(\'https://www.aaechighschools.com/\', \'False\', \'0\', "Home\\n | \\nParent / Student Login \\nCall Today!\\n602.297.8500\\nPrograms\\nAcademics\\nEquine Studies\\nMath & Sciences\\nVeterinary & Medical\\nOur Schools\\nSouth Mountain\\nCalendar\\nParadise Valley\\nCalendar\\nRed Mountain\\nCalendar\\nEstrella Mountain\\nCalendar\\nPrescott Valley\\nCalendar\\nMesa\\nCalendar\\nFor Parents\\nStudent Life\\nAbout AAEC\\nHistory & Philosophy\\nGoverning Board\\nCommunity Relations\\nRegistration\\nOther Scholarship Opportunities\\nResources\\nAccreditation\\nSPED\\nCareers\\nAAEC â\x80\x93 In the News\\nResource Guides\\nPolicies\\nFAQ\\nBlog\\nCommunity Service Events\\nAnnual Financial Reports\\nPre-Enroll\\nContact Us\\nBlog\\nMesa Campus Now Open!\\nAAEC is also building a brand new facility next to MCC in Mesa!!! Call today for details on how you can benefit from AAEC 480-222-3999.\\nJoin Today\\nSpring semester college classes have started for our High School students. Call today for details on how you can benefit from AAEC.\\nPrograms\\nStudents who enroll in our college prep programs can start earning college credits while in high school. \\nParents\\nAAEC is a leading college prep high school with locations in South Mountain, Red Mountain, Estrella Mountain, Paradise Valley and Prescott Valley.\\nSchools\\nArizona Agribusiness & Equine Center (AAEC) is a leading public charter school offering students the opportunity to earn college credits\\nAbout AAEC\\nAAEC high schools offer college preparatory curriculum, and enable students to earn college credits while completing work for their high school diploma.\\nRequest More Info\\n About Our\\n Programs!\\n\\t\\tName (*)\\nInvalid Input\\t\\t\\n\\t\\tEmail (*)\\nInvalid Input\\t\\t\\n\\t\\tPhone\\nInvalid Input\\t\\t\\n\\t\\tCampus\\nSouth Mountain\\nParadise Valley\\nRed Mountain\\nEstrella Mountain\\nPrescott Valley\\nMesa\\nInvalid Input\\t\\t\\nSouth Mountain\\nParadise Valley\\nRed Mountain\\nEstrella Mountain\\nPrescott Valley\\nMesa\\nAcademics\\nEquine Studies\\nMath & Sciences\\nVeterinary & Medical\\n\\r Public Early College High School\\nMission\\nAAEC Early College High School prepares young adults for success now and in the future by promoting lifelong learning through rigorous academic instruction, promoting social responsibility and employability, and providing motivated students with the opportunity to earn college credits while completing their high school requirements.\\nAAEC Early College High School vigorously participates in school-to-career initiatives and occupational education at state and national levels. Please visit the \\nArizona Department of Education\\n website to view AAEC\'s most recent school report card.\\nEnroll in our \\ncollege preparatory classes\\n and start earning college credits today. Contact AAEC Early College High School for information about our \\ncampuses\\n, read our \\nFAQ\\n, or \\nenroll online\\n today.\\nHighly-Qualified Educators Who Make a Difference â\x80¢ Rich Curriculum that Exceeds State Standards â\x80¢ FREE COLLEGE TUITION*\\r\\n*For Qualified Arizona Resident Students\\nPrograms\\nOur Schools\\nFor Parents\\nStudent Life\\nAbout AAEC\\nSouth Mountain\\nParadise Valley\\nRed Mountain\\nEstrella Mountain\\nPrescott Valley\\nResources\\nTax Credit Form\\nPre-Enroll\\nSitemap\\nCareers\\nArizona Agribusiness & Equine Center\\n District Office\\n 3636 N Central Ave Suite 1050\\n Phoenix, AZ 85012\\n ph. 602-297-8500\\n fx. 602-297-8540\\nGoverning Board Meeting Agenda\\nÂ© Copyright 2013 by Arizona Agribusiness & Equine Center. All Rights Reserved")'
# rm = 


# In[1]:


# new_data.columns


# In[7]:


# k = new_data.iloc[7:11]['CMO_WEBTEXT'][0]
# d = '(\'https://'
# s =  [d+e for e in k.split(d) if e]
# #k.split("(\'https://")
# k
# pr = ast.literal_eval(k)
# len(pr)


# In[8]:


# #all_pages is a list of strings of all web page texts
# all_pages = []

# no_nan_data = new_data.loc[:, ['CMO_WEBTEXT']].fillna("")
# col_pages = no_nan_data.loc[1:3]['CMO_WEBTEXT'] #change to new_data['data'] later, but work with first 1 schools for now
# i = 0
# count = 0
# for school_data in col_pages:

#     for tup in school_data:
#         all_pages.append(tup[3])
      

#     i+=1


# In[9]:


# unique_pages_set = set(all_pages)

# #set already get rid of duplicate strings
# ratio_df = pd.DataFrame(np.array(list(unique_pages_set)), columns=["Page"])
# ratio_df["Ratios"] = np.nan
# ratio_df["Similar"] = np.nan #similar will hold similar indexes, > 0.90 ratios

# ratio_df['Ratios'] = ratio_df['Ratios'].astype('object')
# ratio_df['Similar'] = ratio_df['Similar'].astype('object')


# In[32]:


def create_sim_df(pages):
    
    ratio_df = pd.DataFrame(np.array(pages), columns=["Page"])
    ratio_df["Ratios"] = np.nan
    ratio_df["Similar"] = np.nan #similar will hold similar indexes, > 0.90 ratios

    ratio_df['Ratios'] = ratio_df['Ratios'].astype('object')
    ratio_df['Similar'] = ratio_df['Similar'].astype('object')
    
    
    
    if (len(pages) <= 20) :
   
        for a in range(len(pages)):
            sim_ind_list = []
            for b in range(len(pages)):
                #add on every other index execpt the one it's on
                #this way, it'll compare every page to every other page for similarities
                if (a != b):
                    sim_ind_list.append(b)
            ratio_df['Similar'][a] = np.asarray(sim_ind_list)    
   
    else :
        index = 0
        for page0 in pages:
            ratios_list = []
            for page1 in pages:
                ratios_list.append(SeqMatcher(None, page0, page1).ratio())
            ratio_df['Ratios'][index] = ratios_list
            sim_ind_list = np.asarray(np.where((np.asarray(ratios_list) >= 0.7) & (np.asarray(ratios_list) != 1.0))[0]).tolist()
            ratio_df['Similar'][index] = sim_ind_list
            index+=1
                
    num_rows = ratio_df.shape[0] 
    tot_tuples = []

    final_cut_strings = [None] * num_rows
    #count_not_similar = 0  
    #not_sim_list = []
    sim_list = []

    for r in range(num_rows):
        if (len(ratio_df['Similar'][r]) != 0):
            sim_list.append(r)
            #print("has similar > 0.9 at index : " + str(r))
            list_cut = []
            for ind in ratio_df['Similar'][r]:
                list_of_triples = SeqMatcher(None, ratio_df['Page'][r], ratio_df['Page'][ind]).get_matching_blocks()
                zeroth_triple = list_of_triples[0] #first triple, most likely in beginning, most likely a header
                n = zeroth_triple[2] #j to j+n are the indices of the overlapping part
                orig_string = ratio_df['Page'][ind]
                cut_down_string = orig_string[:zeroth_triple[1]] + orig_string[zeroth_triple[1] + n:] #removes overlapping part
                list_cut.append([ind, cut_down_string])
            tot_tuples.extend(list_cut) #list of tuples
        else:
            #print("no similar at index : " + str(r))
            #not_sim_list.append(r)
            #count_not_similar +=1
            final_cut_strings[r]= ratio_df['Page'][r]
    
    return tot_tuples, final_cut_strings, ratio_df




# In[6]:


# index = 0
# for page0 in unique_pages_set:
#     ratios_list = []
#     for page1 in unique_pages_set:
#         ratios_list.append(SeqMatcher(None, page0, page1).ratio())
#     ratio_df['Ratios'][index] = ratios_list
#     sim_ind_list = np.asarray(np.where((np.asarray(ratios_list) > 0.9) & (np.asarray(ratios_list) != 1.0))[0]).tolist()
#     ratio_df['Similar'][index] = sim_ind_list
#     index+=1


# In[7]:


# num_rows = ratio_df.shape[0] 
# tot_tuples = []

# final_cut_strings = [None] * num_rows
# count_not_similar = 0  
# not_sim_list = []
# sim_list = []

# for r in range(num_rows):
#     if (len(ratio_df['Similar'][r]) != 0):
#         sim_list.append(r)
#         #print("has similar > 0.9 at index : " + str(r))
#         list_cut = []
#         for ind in ratio_df['Similar'][r]:
#             list_of_triples = SeqMatcher(None, ratio_df['Page'][r], ratio_df['Page'][ind]).get_matching_blocks()
#             zeroth_triple = list_of_triples[0] #first triple, most likely in beginning, most likely a header
#             n = zeroth_triple[2] #j to j+n are the indices of the overlapping part
#             orig_string = ratio_df['Page'][ind]
#             cut_down_string = orig_string[:zeroth_triple[1]] + orig_string[zeroth_triple[1] + n:] #removes overlapping part
#             list_cut.append([ind, cut_down_string])
#         tot_tuples.extend(list_cut) #list of tuples
#     else:
#         #print("no similar at index : " + str(r))
#         not_sim_list.append(r)
#         count_not_similar +=1
#         final_cut_strings[r]= ratio_df['Page'][r]


# In[8]:


# list_grouped = [[] for x in range(num_rows)] #big list, just put list of strings in spots where needed
# list_indices_of_groups = []
# for ind in range(num_rows):
#     for tup in tot_tuples:
#         if(tup[0] == ind):
#             list_grouped[ind].append(tup[1]) #attach that tuple's string
#             list_indices_of_groups.append(tup[0])


# In[9]:


# #not used
# unique_group_ind = set(list_indices_of_groups) 


# In[10]:


# ind_fill_final_grouped = []
# i = 0
# for group in list_grouped:
#     if (len(group) != 0):
#         #print("list ready to insert at index " + str(i) + "\n")
#         ind_fill_final_grouped.append(i)
    
#     i+=1


# In[11]:


# spot = 0
# #add into final cut strings, the new "cut down" versions of appropriate strings
# for li in list_grouped:
#     if (len(li) != 0):
#         #print(unique_ind_list[spot])
#         final_cut_strings[ind_fill_final_grouped[spot]]= min(li, key=len) #inserts into correct index, what was None before, add in that string now
#         spot+=1


# In[33]:


def create_first_header_cut(tot_tuples, final_cut_strings, num_rows):
    #num_rows = ratio_df.shape[0]
    list_grouped = [[] for x in range(num_rows)] #big list, just put list of strings in spots where needed
    #list_indices_of_groups = []
    for ind in range(num_rows):
        for tup in tot_tuples:
            if(tup[0] == ind):
                list_grouped[ind].append(tup[1]) #attach that tuple's string
                #list_indices_of_groups.append(tup[0])
                
    ind_fill_final_grouped = []
    i = 0
    for group in list_grouped:
        if (len(group) != 0):
            ind_fill_final_grouped.append(i)
    
        i+=1
        
    spot = 0
    #add into final cut strings, the new "cut down" versions of appropriate strings
    for li in list_grouped:
        if (len(li) != 0):
            #print(unique_ind_list[spot])
            final_cut_strings[ind_fill_final_grouped[spot]]= min(li, key=len) #inserts into correct index, what was None before, add in that string now
            spot+=1
    
    return final_cut_strings


# In[13]:


# #first removal of headers in final_cut_Strings currently, but now we want to cut down headers more
# #take out text before the first sentence or text before the first group of 7+ words

# super_final_strings = []
# for s in final_cut_strings:
#     use_punc = False
#     use_sev = False
#     punc = [",", ".", ":", ";"]
#     p_list = []
#     for p in punc:
#         if (s.find(p) != -1):
#             p_list.append(s.find(p))
#         else:
#             p_list.append(len(s))
    
#     punc_ind = min(p_list)
    
#     n_list = [index for index, k in enumerate(s) if k=='\n']
#     start_punc = len(s)
#     for i in n_list:
#         if(i < punc_ind):
#             start_punc = i # start_punc equals the largest index of \n that's less than index of first punctuation
    
#     start = 0
#     end = 0
#     total = ""
#     list_totals = []
#     st_en = []
#     for c in s:
#         if (c not in ['\n', '\t']):
#             total+=(c)
#             end+=1
#         else :
#             if(len(total.split()) >= 7): # we hit 7 words or more, wipe everything before start index
#                 #list_totals.append(total)
#                 st_en.append((start, end))
    
#             total= ""
#             start = end
#     start_sev = len(s)-1 #len(s)-1 #make it huge by default; if there's no group of 7, then start punc will be the smallest
#     if len(st_en) > 0:
#         start_sev = st_en[0][0] #index of first group of words that's >= 7 words; 0th tuple's start value
   

#     #take smaller of the two indices, since we want to use the property which occurs first
#     if start_punc < start_sev:
#         #if start of sentence which ends in/contains puncuation occurs earlier, wipe eveything before that index
#         #only take that index +1 and on, start right after the new line
#         new_string = s[start_punc+1:] 
#         super_final_strings.append(new_string)
        
#     else:
#         #if start of group of words that >= 7 occurs earlier than a sentence with punctuation, wipe eveything before that index
#         #only take that index and on, statr using that begining of the group of 7+ words
#         new_string = s[start_sev:] 
#         super_final_strings.append(new_string)
        
     
            


# In[14]:


def create_second_header_cut(first_header_cut):
    
    super_final_strings = []
    for s in first_header_cut:
        s_new = ""
        if(s is None):
            s_new = ""
        else:
            s_new = s
        use_punc = False
        use_sev = False
        punc = [",", ".", ":", ";"]
        p_list = []
        for p in punc:
            if (s_new.find(p) != -1):
                p_list.append(s_new.find(p))
            else:
                p_list.append(len(s_new))

        punc_ind = min(p_list)

        n_list = [index for index, k in enumerate(s_new) if k=='\n']
        start_punc = len(s_new)
        for i in n_list:
            if(i < punc_ind):
                start_punc = i # start_punc equals the largest index of \n that's less than index of first punctuation

        start = 0
        end = 0
        total = ""
        list_totals = []
        st_en = []
        for c in s_new:
            if (c not in ['\n', '\t']):
                total+=(c)
                end+=1
            else :
                if(len(total.split()) >= 7): # we hit 7 words or more, wipe everything before start index
                    #list_totals.append(total)
                    st_en.append((start, end))

                total= ""
                start = end
        start_sev = len(s_new)-1 #len(s)-1 #make it huge by default; if there's no group of 7, then start punc will be the smallest
        if len(st_en) > 0:
            start_sev = st_en[0][0] #index of first group of words that's >= 7 words; 0th tuple's start value


        #take smaller of the two indices, since we want to use the property which occurs first
        if start_punc < start_sev:
            #if start of sentence which ends in/contains puncuation occurs earlier, wipe eveything before that index
            #only take that index +1 and on, start right after the new line
            new_string = s_new[start_punc+1:] 
            super_final_strings.append(new_string)

        else:
            #if start of group of words that >= 7 occurs earlier than a sentence with punctuation, wipe eveything before that index
            #only take that index and on, statr using that begining of the group of 7+ words
            new_string = s_new[start_sev:] 
            super_final_strings.append(new_string)
        
    return super_final_strings



# In[28]:


#compare between pages of each school

def remove_string_overlaps(tuplist):
   
    unique_tuplist = []
    seen_pages = set() # Initialize list of known pages for a school
    unique_pages=[]
    reversed_pages = []
    tup_indices = []

    cleaned_strings = []
    new_list = []
    
    if (str(tuplist) == 'nan') or (len(tuplist) == 0):
        return new_list
    else :
        
        for tup in tuplist:
            if (tup is not None) and (len(tup) > 3):
                seen_pages.add(tup[3])

        for i in range(len(tuplist)):
            #(tuplist[i][3] is not np.nan) and
            if (len(tuplist[i]) > 3) and (tuplist[i][3] in seen_pages) and (tuplist[i][3]  not in unique_pages):
                unique_tuplist.append(tuplist[i])
                unique_pages.append(tuplist[i][3])
                reversed_pages.append(tuplist[i][3][::-1])
                tup_indices.append(i)
                #print("unique page : " + str(i))

        #now compare all pages with each other 
        #print(unique_tuplist)
        tot_tuples, final_cut_strings, ratio_df = create_sim_df(unique_pages)
        first_header_cut = create_first_header_cut(tot_tuples, final_cut_strings, ratio_df.shape[0])

        #first removal of headers in final_cut_Strings currently, but now we want to cut down headers more
        #take out text before the first sentence or text before the first group of 7+ words  
        second_header_cut = create_second_header_cut(first_header_cut)


        #now run process on reversed strings

        rev_tot_tuples, rev_final_cut_strings, rev_ratio_df = create_sim_df(reversed_pages)

        rev_first_header_cut = create_first_header_cut(rev_tot_tuples, rev_final_cut_strings, rev_ratio_df.shape[0])

        for i in range(len(rev_first_header_cut)):
            #find where to cut the footer off , index

            add_string = second_header_cut[i]
            if i in tup_indices: 
                if(rev_first_header_cut is None):
                    rev_first_header_cut = ""
                if(rev_first_header_cut[i] is None):
                    rev_first_header_cut[i] = ""
                forward_string = rev_first_header_cut[i][::-1]
                #print(forward_string)
                sept = int(len(forward_string)/2)
                half_string = forward_string[len(forward_string) - sept:]
                #find that half in the regular string, and get the end of the half
                #print(type(second_header_cut[i]))
                #print(type(half_string))
                end_index = (second_header_cut[i]).find(half_string) + len(half_string) - 1
                add_string = second_header_cut[i][:end_index]
                #remove the footer aka remove stuff after the end_index
                #keep the stuff before end_index

            cleaned_strings.append(add_string)


        #then iterate through cleaned_strings and inset into each tuple

        for count in range(len(cleaned_strings)):
            new_tup = (tuplist[tup_indices[count]][0], tuplist[tup_indices[count]][1], tuplist[tup_indices[count]][2], cleaned_strings[count])
            new_list.append(new_tup)
    #         print(tup_indices[count])
    #         print(cleaned_strings[count])
    #         print("\n  \n")

    #     print(len(new_list))
    #     print(len(tup_indices))
        return new_list


#first make a list of tuples that has no nan values and has
#if not np.isnan(tuplist):   CHECK FOR NAN OUTSIDE

#k = remove_string_overlaps(new_data['WEBTEXT'][11])





# In[2]:


# index = 0
# ki = []
# lengths = []
# new_data['WEBTEXT'] = new_data['WEBTEXT'].fillna("")
# for li in new_data['WEBTEXT']:
#     if len(li) <100:
#         ki.append(index)
        
#         #print(index)
#     lengths.append(len(li))
#     index+=1
# print(len(ki))
# print(len(new_data['WEBTEXT']))
# print(np.average(lengths))
# print(np.median(lengths))
# ki


# In[3]:


# lengths


# In[4]:


# print(set(lengths))


# In[5]:


# twenty_count = 0
# for li in new_data['WEBTEXT']:
#     if len(li) <=20:
#         twenty_count +=1

# print(twenty_count/len(new_data['WEBTEXT']))


# In[6]:


# k = remove_string_overlaps(new_data['WEBTEXT'][9])


# In[7]:


# k


# In[8]:


# r = sorted(lengths, key=int)  
# np.median(r)
# for i in r:
#     print(i)


# In[36]:


#apply remove_string_overlaps on each school, aka on each row of new_data
#since pages of a school will likely be similar to the other pages within that school own

def parse_df(old_list):
    
    new_list = remove_string_overlaps(old_list)
    return new_list
    


# In[37]:


new_data['WEBTEXT'] = new_data['WEBTEXT'].fillna("0").apply(ast.literal_eval)


arr_of_dfs = np.array_split(new_data, len(new_data['WEBTEXT']))

global merged_df_file
merged_df_file = folder_prefix+"merged_df_WEBTEXT.csv" # Prepare file name


# In[38]:


# tqdm.pandas(desc="Processing:")

# arr_of_dfs[0]['WEBTEXT'] = arr_of_dfs[0]['WEBTEXT'].progress_apply(parse_df)

# arr_of_dfs[0].to_csv(merged_df_file, mode="w", index=False, header=arr_of_dfs[0].columns.values, sep="\t", encoding="utf-8")


#df = pd.read_csv(folder_prefix+'_mergedf_WEBTEXT.csv', header=arr_of_dfs[0].columns.values, sep="\t", encoding="utf-8")


# In[43]:






# In[45]:


def chunk_assign(df_chunk):
    global num
    
    need_clean_chunk = df_chunk.loc[df_chunk['OVERLAPS_REMOVED'] == 0]
    
    #already_cleaned_chunk = df_chunk.loc[df_chunk['OVERLAPS_REMOVED'] == 1] don't do anything with this
    
    if (need_clean_chunk.shape[0] > 0): #there are actually rows which haven't been parsed yet
        need_clean_chunk['WEBTEXT'] = need_clean_chunk['WEBTEXT'].apply(parse_df)
        need_clean_chunk['OVERLAPS_REMOVED'] = 1
        
        if num==0: # Save first slice to new file (overwriting if needed)
            need_clean_chunk.to_csv(folder_prefix + "parsed_df_5.csv", mode="w", index=False, header=df_chunk.columns.values, sep="\t", encoding="utf-8")
        
        else:
            need_clean_chunk.to_csv(folder_prefix + "parsed_df_5.csv", mode="a", index=False, header=False, sep="\t", encoding="utf-8")

    
#     curr_final_df = pd.read_csv(merged_df_file , sep="\t", low_memory=False, encoding="utf-8")
#     print()
    
    num+=1
        #final_chunk = pd.concat([need_clean_chunk, already_cleaned_chunk])


    
    
    #logging.info((need_clean_chunk.shape[0] / df_chunk.shape[0]) * 100)
    #print(num * 44)
    
    #free chunk?
   # logging.info("df chunk saved to " + df_filepath )
    
    return df_chunk


# In[46]:


num = 0
#start_time = time.time()
#tqdm.pandas(desc="Processing:")
#tqdm.pandas(desc="Processing all: " )
orig_num_rows = new_data.shape[0] 

# curr_merged_df = pd.read_csv(merged_df_file , sep="\t", low_memory=False, encoding="utf-8")
# merged_num_rows = curr_merged_df.shape[0]


numcpus = len(os.sched_getaffinity(0)) # Detect and assign number of available CPUs
p = mp.Pool(numcpus)
result_df = p.map(chunk_assign, arr_of_dfs)

starttime=time.time()

# while True:   
#     second_df = pd.read_csv(folder_prefix + "parsed_df_2.csv", sep="\t", low_memory=False, encoding="utf-8")
#     second_num_rows = second_df.shape[0]
#     sum_so_far = merged_num_rows + second_num_rows
#     diff = orig_num_rows - sum_so_far
#     logging.info(sum_so_far)
#     print("Overlap == 0 row : " + str(second_num_rows)+ " . Total first + second num rows : " + str(sum_so_far) +  " . Num rows to go : " + str(diff))
#     time.sleep(120.0 - ((time.time() - starttime) % 120.0))





p.close()

#print("--- %s seconds ---" % (time.time() - start_time))


# In[113]:


# #list of common words in footers
# footer_list = ["Copyright", "All Rights Reserved",  "Read More", 
#                "Useful Links", "Search", "Survey", "Feed", "Fax", "Address",  "Sitemap", 
#               "Jobs"]
# #facebook, contact us, enroll etc occurs in headers as well, so had to take that out

# footers_removed_strings = []

# #look for the keyword in each string and if found, remove all the text after it
# for s in super_final_strings:
#     no_newline = s.replace("\n", " ")
#     new_list = []
#     for word in footer_list:
#         if (no_newline.find(word) != -1):
#             new_list.append(no_newline.find(word))
#         else :
#             new_list.append(len(s))

#     #get the index of the earliest occurence of a keyword
#     f_ind = min(new_list)
    
#     #go back to new line or period right before and wipe out everything after that
#     n_list = [index for index, k in enumerate(s) if k in ['\n', '.']]
#     start_punc = len(s)
#     for i in n_list:
#         if(i < f_ind):
#             start_punc = i # start_punc equals the largest index of \n or . that's less than index of the keyword
            
#     if start_punc < f_ind:
#             footer_rem = s[:start_punc]
#             footers_removed_strings.append(footer_rem) 
#     else:
#             footer_rem = s[:f_ind]
#             footers_removed_strings.append(footer_rem) 
              


# In[114]:


# kr = super_final_strings[4]
# no_newline = kr.replace("\n", " ")
# new_list = []
# li = []
# for word in footer_list:
    
#     if (no_newline.find(word) != -1):
#         new_list.append(no_newline.find(word)) #index at which the word starts
#     else :
#         new_list.append(len(kr))

# #get the index of the earliest occurence of a keyword
# f_ind = min(new_list)
    
# #go back to new line or period right before and wipe out everything after that
# n_list = [index for index, k in enumerate(kr) if k in ['\n', '.']]
# start_punc = len(kr)
# for i in n_list:
#     if(i < f_ind):
#         start_punc = i # start_punc equals the largest index of \n or . that's less than index of the keyword
            
# if start_punc < f_ind: #if punctuation occurs before keyword, wipe out everything after punctuation
#         footer_rem = kr[:start_punc]
#         li.append(footer_rem) 
# else:
#         footer_rem = kr[:f_ind] #else just wipe out everything after punctuation
#         li.append(footer_rem) 
# print(word + ": " + "start_punc : " + str(start_punc) + " . f_ind : " + str(f_ind))

# li[0]


# In[115]:


#kr[646:]


# In[116]:


#footers_removed_strings are the final, most cut down versions of the web text
#we did process on "WEBTEXT" column of new_data
#now repeat for "CMO_WEBTEXT" column of new_data


# In[9]:


# footers_removed_strings


# In[10]:


# for count in range(len(footers_removed_strings)):
#     if "\nPre-enroll" in footers_removed_strings[len(footers_removed_strings[count])-13:]:
#         print(str(count))
        


# In[11]:


#ratio_df['Page'][4]


# In[12]:


#footers_removed_strings[1]


# In[13]:


#super_final_strings[1]

