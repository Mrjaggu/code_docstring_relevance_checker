import re
from stop_words import get_stop_words

STOP_WORDS = get_stop_words('english')

# some custom stopwords depending on the analysis what we captured
custom_stowords_list = ['def','self','returns','return',
                        'zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz',
                       'zzzzzzzzzz','zzzzzzzz','zzzzzzz','zzzzz',
                       'aaaabnzacyceaaaadaqabaaabaqddurxdpyhoquyodutxjduzqmjyjqjszt',
       'aaaabnzacyceaaaadaqabaaabaqdcgxehzf', 'aaaabnzacyceaaa',
       'aaaabnzacyce', 'aaaabnzacyc', 'aaaabnzacy',
       'aaaabnzackcmaaacbakdpkarimlm', 'aaaabbbccdaabbb', 'aaaabbbcca',
       'aaaabaaacaaadaaaeaaafaaagaaahaaaiaaajaaakaaalaaamaaanaaaoaaapaaaqaaaraaasaaataaa',
       'aaaaargh', 'aaaaabaaa', 'aaaaabaa',
       'aaaaaaeceeeeiiiidnoooooouuuuysaaaaaaaceeeeiiii', 'aaaaaabaedaa',
       'aaaaaaaarge', 'aaaaaaaalaaaaaaadwpwaaaaaaaaaa',
       'aaaaaaaalaaaaaaadwpqzmzmzmdk', 'aaaaaaaadjuqwsqicqdwlclimq',
       'aaaaaaaaaaagaagaaagaaa', 'aaaaaaaaaaaahaaaaaaaaa',
       'aaaaaaaaaaaaaaaaaadwpwaaaaaaapc',
       'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
       'aaaaaaaaaaaaaaaa', 'aaaaaaaaaaaaaaa', 'aaaaaaaaaaaaaa',
       'aaaaaaaaaaaaa', 'aaaaaaaaaaaa', 'aaaaaaaaaaa', 'aaaaaaaaaa',
       'aaaaaaaaa', 'aaaaaaaa', 'aaaaaaa', 'aaaaaa', 'aaaaa', 'aaaa',
       'aaa', 'aa']

STOP_WORDS.extend(custom_stowords_list)


def get_method_name(code:str)->str:
    """Extracting method name
     from method name as a feature
     """
    
    return code.split(":")[0]

def preprocess(text):
    """removing stopwords and numeric from string """

    text = str(text).lower()
    text = " ".join([words for words in text.split() if words.lower() not in STOP_WORDS or \
            words.lower() not in custom_stowords_list])        
    text = re.sub(r"([0-9]+)","", text)

    return text