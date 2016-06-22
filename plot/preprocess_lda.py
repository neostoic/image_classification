import glob
import re

import pandas as pd

input_path = r'C:\Users\crobe\Google Drive\DataMiningGroup\Results\LSTM\original'
input_files = glob.glob(input_path + r"\*.log")
cols = ['seq_length', 'batch_sz', 'embedding_sz', 'lr', 'iteration', 'loss_tr', 'norm_tr', 'loss_val', 'conf_avg',
        'conf_med', 'bleu-4']
result_df = pd.DataFrame({}, columns=cols)
idx = 0
for result_file in input_files:
    with open(result_file, 'r') as fin:
        for line in fin:
            # print line
            search_config = re.search(
                'Seq_length: (\d+), Batch_size: (\d+), Embedding_size: (\d+), Learning_rate: (\d+.\d+)', line)
            search_values = re.search('INFO : (\d+)\t([^\t]*)\t([^\t]*)\t([^\t]*)\t([^\t]*)\t([^\t]*)\t([^\t]*)', line)
            if search_config:
                seq_length = int(search_config.group(1))
                batch_sz = int(search_config.group(2))
                embedding_sz = int(search_config.group(3))
                lr = float(search_config.group(4))
                config = [seq_length, batch_sz, embedding_sz, lr]
                print seq_length, batch_sz, embedding_sz, lr
            if search_values:
                iteration = int(search_values.group(1))
                loss_tr = float(search_values.group(2))
                norm_tr = float(search_values.group(3))
                loss_val = float(search_values.group(4))
                conf_avg = float(search_values.group(5))
                conf_med = float(search_values.group(6))
                bleu = float(search_values.group(7).strip('\n'))
                values = [iteration, loss_tr, norm_tr, loss_val, conf_avg, conf_med, bleu]
                result_df.loc[idx] = config+values
                idx += 1

result_df.to_csv(r'C:\Users\crobe\Google Drive\DataMiningGroup\Results\LSTM\original\results.csv')
result_df.to_excel(r'C:\Users\crobe\Google Drive\DataMiningGroup\Results\LSTM\original\results.xlsx')
