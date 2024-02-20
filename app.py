import streamlit as st

# Set the page layout
st.set_page_config(page_title="GeekMAN", layout="centered")

# # Inject custom CSS
# def inject_custom_css():
#     custom_css = """
#     <style>
#         /* Target the text input directly */
#         .stTextInput>div>div>input {
#             background-color: #dbdce4; /* Background color */
#             font-weight: bold;
#         }
#         .stButton>button {
#             background-color: #dbdce4;
            
#         }
#     </style>
#     """
#     st.markdown(custom_css, unsafe_allow_html=True)

# inject_custom_css()

####################
####################
###GeekMANCode######
####################
####################
import pandas as pd
import sys
import string
import bisect
import re
from py_stringmatching.similarity_measure.levenshtein import Levenshtein
from py_stringmatching.similarity_measure.monge_elkan import MongeElkan
from py_stringmatching.similarity_measure.jaccard import Jaccard
import glob
import time


# dictionary file path
dictionary_filepath = 'resources/curated_word_list.txt'
username_1, username_2 = 'username_1', 'username_2'
sim_score = 'sim_score'

@st.cache_data
def read_initial_conf():
    
    word_list = set();
    with open(dictionary_filepath) as file:
        lines = file.readlines()

        for line in lines:
            word_list.add(line.strip())
    print('number of words/phrases in the dictionary', len(word_list))

    alphabet_lower = list(string.ascii_lowercase)
    alphabet_upper = list(string.ascii_uppercase)
    digits = list(string.digits)
    lines = [
            '0od',
            '1iltj',
            '2z',
            '3es',
            '4ar',
            '5s',
            '6g',
            '7tljr',
            '8b',
            '9g',
            '@a',
            '!i',
            '$s',
        ]
    slangification_map = {}
    for line in lines:
        slangification_map[line[0]] = line[1:]
    slang_alphabet = list(slangification_map.keys())  
    print(alphabet_lower[:5], alphabet_upper[:5], digits[:5], slang_alphabet)
    
    
    return {
        'word_list': word_list,
        'alphabet_lower': alphabet_lower,
        'alphabet_upper': alphabet_upper,
        'digits': digits,
        'slangification_map': slangification_map,
        'slang_alphabet': slang_alphabet,
        'alphabet_lower': alphabet_lower,
        'alphabet_upper': alphabet_upper,
    }



initial_conf = read_initial_conf()

# configurable
min_length = 3

word_list = initial_conf['word_list'];
alphabet_lower = initial_conf['alphabet_lower'];
alphabet_upper = initial_conf['alphabet_upper'];
digits = initial_conf['digits'];
slangification_map = initial_conf['slangification_map'];
slang_alphabet = initial_conf['slang_alphabet']; 

lower_1, lower_2 = 'lower_1', 'lower_2'
w_o_sym_dig_1, w_o_sym_dig_2 = 'w_o_sym_dig_1', 'w_o_sym_dig_2'
cap_ltr_1, cap_ltr_2 = 'cap_ltr_1', 'cap_ltr_2'
sym_1, sym_2 = 'sym_1', 'sym_2'
dig_1, dig_2 = 'dig_1', 'dig_2'
dict_1, dict_2 = 'dict_1', 'dict_2'  

class WeightedIntervalScheduling(object):
    def __init__(self, I):
        self.I = sorted(I, key=lambda tup: tup[1])  # (key = lambda tup : tup[1])
        self.OPT = []
        self.solution = []

    def previous_intervals(self):
        start = [task[0] for task in self.I]
        finish = [task[1] for task in self.I]
        p = []

        for i in range(len(self.I)):
            # finds idx for which to input start[i] in finish times to still be sorted
            idx = bisect.bisect(finish, start[i]) - 1
            p.append(idx)

        return p

    def find_solution(self, j):
        if j == -1:
            return

        else:
            if (self.I[j][2] + self.compute_opt(self.p[j])) > self.compute_opt(j - 1):
                self.solution.append(self.I[j])
                self.find_solution(self.p[j])

            else:
                self.find_solution(j - 1)

    def compute_opt(self, j):
        if j == -1:
            return 0

        elif (0 <= j) and (j < len(self.OPT)):
            return self.OPT[j]

        else:
            return max(
                self.I[j][2] + self.compute_opt(self.p[j]), self.compute_opt(j - 1)
            )

    def weighted_interval(self):
        if len(self.I) == 0:
            return 0, self.solution

        self.p = self.previous_intervals()

        for j in range(len(self.I)):
            opt_j = self.compute_opt(j)
            self.OPT.append(opt_j)

        self.find_solution(len(self.I) - 1)

        return self.OPT[-1], self.solution[::-1]

    
def get_dict_chunks(username):

    
    n = len(username)
    next_list = [(-1, -1, -1) for _ in range(n)]
    

    i = n-1
    while i >= 0:

        next_list[i] = (i, i + 1, 1)
        

        if i + min_length <=n and username[i] in alphabet_lower:
            
            local_optimal = 0
            
            curr_length = min_length
            for end in range(i + min_length -1, n):
                
                if username[i: i+curr_length] in word_list:
                    

                    next_length = 0
                    if i + curr_length < n:
                        
                        next_end = next_list[i + curr_length][1]
                        next_length = next_end - (i+curr_length)
                    
                    if curr_length + next_length >= local_optimal:
                        local_optimal = curr_length + next_length
                        next_list[i] = (i, i + curr_length, curr_length)
                
                curr_length += 1

        i = i-1

        
    weightedinterval = WeightedIntervalScheduling(next_list)
    _, chunks = weightedinterval.weighted_interval()
    
    
    dict_chunks = [chunk[2] for chunk in chunks if chunk[2] > 1]
    dict_chunk_count = len(dict_chunks)
    coverage = sum(dict_chunks)
    
    return (dict_chunk_count, chunks, coverage)

    
def slangify(username):
    
    slang_chars_found = []
    for char in slangification_map:
        
        if char in username:
            slang_chars_found.append(char)
    
    transformed_usernames = []
    if len(slang_chars_found):
        
        transformed_usernames.append(username)
        for slang_char in slang_chars_found:
            
            new_transformed_usernames = []
            replace_chars = slangification_map[slang_char]
            for replace_char in replace_chars:
                
                for transformed_username in transformed_usernames:
                    new_transformed_usernames.append(\
                        transformed_username.replace(slang_char, \
                            replace_char))
            transformed_usernames = new_transformed_usernames
            
    return set(transformed_usernames)


def get_dict_token_based_chunkification(username):
    
    if username.isnumeric():
        return []
    
    preprocess = lambda uname : ''.join([char for char in list(uname.lower()) if char in set(alphabet_lower + slang_alphabet)])
    username = preprocess(username)
    
    chunkification_no_slang = get_dict_chunks(username)
    token_based_chunkification = [username[chunk[0]: chunk[1]] \
             for chunk in chunkification_no_slang[1] ]
    
    slangified_usernames = slangify(username)
    opt_chunkification_slang = chunkification_no_slang 
    opt_slangified_username = None
    if len(slangified_usernames):
        
        for slangified_username in slangified_usernames:
            
            chunkification_slang = get_dict_chunks(slangified_username)
            
            optimal_cond = \
                (chunkification_slang[2] > opt_chunkification_slang[2]) or \
                (chunkification_slang[2] == opt_chunkification_slang[2] and \
                    chunkification_slang[0] < opt_chunkification_slang[0]) 
            
            if  optimal_cond:
                opt_chunkification_slang = chunkification_slang
                opt_slangified_username = slangified_username
        
        if opt_slangified_username:
            token_based_chunkification = \
            [opt_slangified_username[chunk[0]: chunk[1]] \
                 if chunk[2] > 1 else username[chunk[0]: chunk[1]] \
                     for chunk in opt_chunkification_slang[1] ]
    
    return [token for token in token_based_chunkification if len(token) >= min_length]



def get_digit_based_chunkification(username):
    
    username = username.lower()
    letter_groups = re.findall("([^0-9]+)", username)
    digit_groups = re.findall("([0-9]+)", username)
    
    letter_groups = [ chunk.strip() for chunk in letter_groups ]
    digit_groups = [ chunk.strip() for chunk in digit_groups ]
    
    return (letter_groups, digit_groups)


def get_symbol_based_chunkification(username):
    
    username = username.lower()
    chunks = re.split('[^a-zA-Z0-9]+', username) 
    
    return [ chunk for chunk in chunks if len(chunk) > 0]


def get_capital_letter_based_chunkification(username):
    
    if username.islower():
        return []
    
    start = 0
    chunks = []
    for i in range(1, len(username)):
        
        if username[i] in alphabet_upper:
            chunks.append(username[start:i].lower().strip())
            start = i
    
    chunks.append(username[start:len(username)].lower())
    
    return chunks


def get_username_without_symbol_digit(username):
    
    username = username.lower()
    chunks = re.split('[^a-z]+', username)
    
    return ''.join(chunks)

def get_chunkification_length(chunks):
    
    return sum([len(chunk) for chunk in chunks])


def create_column_value_for_list_attrs(list_vals):
    
    return ','.join(list_vals)


def get_similarity_score(pair_row):
    
    scores = []
    monge_elkan_method = MongeElkan(sim_func=Levenshtein().get_sim_score)
    levenshtein_method = Levenshtein()
    
    ########### username_lower
    src_lower_len, target_lower_len = len(pair_row[lower_1]), len(pair_row[lower_2])
    if src_lower_len == 0 or target_lower_len == 0:
        score_on_username_lower = 0.0
    
    else:
        score_on_username_lower = levenshtein_method.get_sim_score(pair_row[lower_1], 
                                                              pair_row[lower_2])
    scores.append(score_on_username_lower)
    
    ########### without_sym_dig
    src_feature_len, target_feature_len = len(pair_row[w_o_sym_dig_1]), len(pair_row[w_o_sym_dig_2])
    if src_feature_len == 0 or target_feature_len == 0:
        score_on_without_sym_dig = 0.0
    
    else:
        score_on_without_sym_dig = levenshtein_method.get_sim_score(pair_row[w_o_sym_dig_1], 
                                                              pair_row[w_o_sym_dig_2])
        
#         weight = (src_feature_len + target_feature_len)/(src_lower_len + target_lower_len)
#         score_on_without_sym_dig = score_on_without_sym_dig * weight
        
    scores.append(score_on_without_sym_dig)
    
    ########### cap_chunks
    chunk_list_1, chunk_list_2 = pair_row[cap_ltr_1].split(','), pair_row[cap_ltr_2].split(',')
    src_feature_len, target_feature_len = get_chunkification_length(chunk_list_1), \
                                            get_chunkification_length(chunk_list_2)
    
    if src_feature_len == 0 or target_feature_len == 0:
        
        score_on_cap_chunks = 0.0
    else:
        if len(chunk_list_1) < len(chunk_list_2):
            chunk_list_1, chunk_list_2 = chunk_list_2, chunk_list_1
        score_on_cap_chunks = Jaccard().\
                                    get_raw_score(chunk_list_1, chunk_list_2)
        
#         weight = (src_feature_len + target_feature_len)/(src_lower_len + target_lower_len)
#         score_on_cap_chunks = score_on_cap_chunks * weight
        
    scores.append(score_on_cap_chunks)
    
    ########### sym_chunks
    chunk_list_1, chunk_list_2 = pair_row[sym_1].split(','), pair_row[sym_2].split(',')
    src_feature_len, target_feature_len = get_chunkification_length(chunk_list_1), \
                                            get_chunkification_length(chunk_list_2)
        
    if src_feature_len == 0 or target_feature_len == 0:
        
        score_on_sym_chunks = 0.0
    else:
        if len(chunk_list_1) < len(chunk_list_2):
            chunk_list_1, chunk_list_2 = chunk_list_2, chunk_list_1
        score_on_sym_chunks = monge_elkan_method.get_raw_score(chunk_list_1, chunk_list_2)
        
#         weight = (src_feature_len + target_feature_len)/(src_lower_len + target_lower_len)
#         score_on_sym_chunks = score_on_sym_chunks * weight
        
    scores.append(score_on_sym_chunks)
    
    ########### dig_chunks
    chunk_list_1, chunk_list_2 = pair_row[dig_1].split(','), pair_row[dig_2].split(',')
    src_feature_len, target_feature_len = get_chunkification_length(chunk_list_1), \
                                            get_chunkification_length(chunk_list_2)
        
    if src_feature_len == 0 or target_feature_len == 0:
        
        score_on_dig_chunks = 0.0
    else:
        if len(chunk_list_1) < len(chunk_list_2):
            chunk_list_1, chunk_list_2 = chunk_list_2, chunk_list_1
        score_on_dig_chunks = monge_elkan_method.get_raw_score(chunk_list_1, chunk_list_2)
        
#         weight = (src_feature_len + target_feature_len)/(src_lower_len + target_lower_len)
#         score_on_dig_chunks = score_on_dig_chunks * weight
        
    scores.append(score_on_dig_chunks)
    
    
    ########### dict_token_chunks
    chunk_list_1, chunk_list_2 = pair_row[dict_1].split(','), pair_row[dict_2].split(',')
    src_feature_len, target_feature_len = get_chunkification_length(chunk_list_1), \
                                            get_chunkification_length(chunk_list_2)
        
    if src_feature_len == 0 or target_feature_len == 0:
        
        score_on_dict_token_chunks = 0.0
    else:
        if len(chunk_list_1) < len(chunk_list_2):
            chunk_list_1, chunk_list_2 = chunk_list_2, chunk_list_1
        score_on_dict_token_chunks = monge_elkan_method.get_raw_score(chunk_list_1, chunk_list_2)
        
        weight = (src_feature_len + target_feature_len)/(src_lower_len + target_lower_len)
        score_on_dict_token_chunks = score_on_dict_token_chunks * weight
        
    scores.append(score_on_dict_token_chunks)
    
#     print(scores)
    
    return max(scores)

@st.cache_data
def get_matching_score(df):  

    df[username_1] = df[username_1].astype(str)
    df[username_2] = df[username_2].astype(str)

    print('a few of the username pairs')
    print(df.head(5))
    print()

    #### create features

    # findLower
    df[lower_1] = df.apply(lambda row: row.username_1.lower(), axis = 1)
    df[lower_2] = df.apply(lambda row: row.username_2.lower(), axis = 1)

    # findUsernameWithoutSymbolDigit
    df[w_o_sym_dig_1] = df.apply(lambda row: get_username_without_symbol_digit(row.username_1), axis = 1)
    df[w_o_sym_dig_2] = df.apply(lambda row: get_username_without_symbol_digit(row.username_2), axis = 1)

    # findCapitalLetterBasedChunkification
    df[cap_ltr_1] = df.apply(lambda row: create_column_value_for_list_attrs(\
                                            get_capital_letter_based_chunkification(row.username_1)), axis = 1)
    df[cap_ltr_2] = df.apply(lambda row: create_column_value_for_list_attrs(\
                                            get_capital_letter_based_chunkification(row.username_2)), axis = 1)


    # findSymbolBasedChunkification        
    df[sym_1] = df.apply(lambda row: create_column_value_for_list_attrs(\
                                            get_symbol_based_chunkification(row.username_1)), axis = 1)
    df[sym_2] = df.apply(lambda row: create_column_value_for_list_attrs(\
                                            get_symbol_based_chunkification(row.username_2)), axis = 1)

    # findDigitBasedChunkification
    df[dig_1] = df.apply(lambda row: create_column_value_for_list_attrs(\
                                            get_digit_based_chunkification(row.username_1)[0]), axis = 1)
    df[dig_2] = df.apply(lambda row: create_column_value_for_list_attrs(\
                                            get_digit_based_chunkification(row.username_2)[0]), axis = 1)

    # findDictionaryTokenBasedChunkification        
    df[dict_1] = df.apply(lambda row: create_column_value_for_list_attrs(\
                                            get_dict_token_based_chunkification(row.username_1)), axis = 1)
    df[dict_2] = df.apply(lambda row: create_column_value_for_list_attrs(\
                                            get_dict_token_based_chunkification(row.username_2)), axis = 1)

    # find similarity
    df[sim_score] = df.apply(lambda row: get_similarity_score(row), axis = 1)
    df[sim_score] = df[sim_score].round(3)

    print("matching is successful")
    
    df = df[[username_1, username_2, sim_score]]
    print(df.head(5))
    return df


####################
####################
###GeekMANCode######
####################
####################


####################
@st.cache_data
def read_file(uploaded_file):
    
    with st.spinner('File is being loaded. Wait for a while'):
        time.sleep(1)
        df = pd.read_csv(uploaded_file, names=[username_1, username_2])
        print('total number of usernames', df.shape)
    
    return df


####################


# Function to display UI for the "first" option
def display_first_option():
    # Display two text inputs with titles and placeholders
    
    
    text_input1 = st.text_input("**Username 1**", placeholder="Anon-Exploiter", value="Anon-Exploiter")
    text_input2 = st.text_input("**Username 2**", placeholder="An0n3xpl0it3r", value="An0n3xpl0it3r")
    
    if st.button("**Find Similarity Score**", key="action1"):
        
        try:
            with st.spinner('Similarity is being estimated, wait for 1-10 seconds'):
                time.sleep(1)
                df = pd.DataFrame([{username_1: text_input1, username_2: text_input2}])
                df = get_matching_score(df)
                score = df.to_dict(orient='records')[0]['sim_score']
            t = st.write(f"###### Similarity Score is: {score}")  # Display text below the button
        except Exception as e:
            print(e)
            st.write("Exception occured !! Please try again.")

# Function to display UI for the "second" option
def display_second_option():
    url = 'https://github.com/mrayhanulmasud/geekman/blob/main/data/test.csv'
    
    uploaded_file = st.file_uploader("**Upload a CSV file with list of username pairs** [Example file format]({url})", type="csv")

    if uploaded_file is not None:
#         df = pd.read_csv(uploaded_file)
        try:
            
            df = read_file(uploaded_file)

            if st.button("**Find Similarity Score**"):

                with st.spinner('Similarity is being estimated, wait for 10-20 seconds'):
                    # Assuming the CSV has columns 'val1', 'val2', and 'score'
                    time.sleep(3)
                    df = get_matching_score(df)
                st.download_button("**Download Results in CSV**", df.to_csv(index=False), "file.csv", "text/csv", key='download-csv')
                st.table(df)  # Display the dataframe as a table
                
        except Exception as e:
            print(e)
            st.write("### Exception occured !! Please try again.")

# Header and Footer
st.header("GeekMAN - Geek Oriented Username Matching across Online Networks")
tab1, tab2 = st.tabs([f"**Regular Search**", "**File Upload**"])

with tab1:
   display_first_option()

with tab2:
   display_second_option()

# Footer
# Use markdown with unsafe_allow_html=True to render HTML
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f1f1f1;
        color: black;
        text-align: center;
        padding: 10px;
    }
    </style>
    <div class="footer">
      <p>MAVERICS © 2024 GeekMAN</p>
    </div>
    """,
    unsafe_allow_html=True
)


