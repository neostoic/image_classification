import glob
import re

import pandas as pd

input_path = r'C:\Users\crobe\Google Drive\DataMiningGroup\Results\LDA\second'
input_files = glob.glob(input_path + r"\*.log")
cols = ['num_topics_val', 'iterations', 'passes', 'alpha', 'eta', 'decay', 'offset', 'per_word_bound', 'perpx',
        'hold_docs', 'hold_words']

config_re_str = r'Batch_LDA (\d+) (\d+) (\d+) ([^ ]*) ([^ ]*) (\d+.\d+) (\d+.\d+)'
perpx_re_str = r'INFO : (-\d+.\d+) [^\d]*(\d+.\d+) [^\d]*(\d+) [^\d]*(\d+)'

result_df = pd.DataFrame({}, columns=cols)
idx = 0
for result_file in input_files:
    with open(result_file, 'r') as fin:
        for line in fin:
            # print line
            search_config = re.search(config_re_str, line)
            search_values = re.search(perpx_re_str, line)
            if search_config:
                num_topics = int(search_config.group(1))
                iterations = int(search_config.group(2))
                passes = int(search_config.group(3))
                alpha = search_config.group(4)
                eta = search_config.group(5)
                decay = float(search_config.group(6))
                offset = float(search_config.group(7))

                config = [num_topics, iterations, passes, alpha, eta, decay, offset]
                print config
            if search_values:
                per_word_bound = float(search_values.group(1))
                perpx = float(search_values.group(2))
                hold_docs = int(search_values.group(3))
                hold_words = int(search_values.group(4))
                values = [per_word_bound, perpx, hold_docs, hold_words]
                result_df.loc[idx] = config + values
                idx += 1

result_df.to_csv(r'C:\Users\crobe\Google Drive\DataMiningGroup\Results\LDA\second\results.csv')
result_df.to_excel(r'C:\Users\crobe\Google Drive\DataMiningGroup\Results\LDA\second\results.xlsx')
