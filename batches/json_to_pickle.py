# Script to pickle json data automatically
# Should only work with tables created with Apache drill since they have a different format
# This code was adapted from a post from user 'tback' on  to the
# stackoverflow forum here:  http://stackoverflow.com/a/8730674
#
# Usage:
# 
# python path\to\json_to_pickle.py path\to\json_file.json
import json
import os
import re
import sys

from pandas.io.json import json_normalize

# Create regex expression to properly parse JSON file created by Apache Drill.
FLAGS = re.VERBOSE | re.MULTILINE | re.DOTALL
WHITESPACE = re.compile(r'[ \t\n\r]*', FLAGS)


# ConcatJSONDecoder subclasses the original decoder, to parse Apache Drill JSON
class ConcatJSONDecoder(json.JSONDecoder):
    def decode(self, s, _w=WHITESPACE.match):
        s_len = len(s)

        objs = []
        end = 0
        # Parse the JSON file removing whitespace with REGEX expression
        while end != s_len:
            obj, end = self.raw_decode(s, idx=_w(s, end).end())
            end = _w(s, end).end()
            objs.append(obj)
        return objs


# Accept system arguments
if len(sys.argv) > 1:
    # First argument should be the file path of the JSON file
    filename = sys.argv[1]
    # The pickled file will have the same name as the original, replacing .json with .pkl
    pkl_file = os.path.splitext(filename)[0] + '.pkl'
    # Load the JSON data using the new decoder
    with open(filename) as f:
        json_data = json.load(f, cls=ConcatJSONDecoder)
    # Use pandas normalize function to obtain a pandas dataframe (dictionary-style)
    df = json_normalize(json_data)
    # Pickle pandas data frame
    df.to_pickle(pkl_file)
    print 'JSON data saved as' + pkl_file

# If no arguments are introduced, try with the path defined below
else:
    filename = 'C:\path\to\json'
    pkl_file = os.path.splitext(filename)[0] + '.pkl'
    with open(filename) as f:
        json_data = json.load(f, cls=ConcatJSONDecoder)
    df = json_normalize(json_data)
    df.to_pickle(pkl_file)
    print 'JSON data saved as' + pkl_file
