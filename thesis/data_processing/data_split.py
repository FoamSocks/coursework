import pandas as pd
import sys
from datetime import datetime
from os import makedirs

DATA_DIR = '/data/alexander.huang/data/0903_data/'

start_time = datetime.now()

# inbound or outbound
direction = sys.argv[1]
# train, test, or val
dataset = sys.argv[2]

# added _drop folder after dropping data samples with OOV ports/protocols
SAVE_DIR = DATA_DIR + dataset+'_drop/'

makedirs(SAVE_DIR, exist_ok=True)

# import data
print('Loading', direction, dataset, 'dataset...')
path = DATA_DIR + direction + '_proc_' + dataset + '_drop.parq'
print('path:', path)
df = pd.read_parquet(path)

print(direction, dataset, 'data label counts:')
print(df['packet_label'].value_counts())
print()
print('loaded data:')
print(df.head(3))
sys.stdout.flush()
print('\nLoad complete.')
sys.stdout.flush()

if(dataset=='train' and direction=='inbound'):
    df = df.sample(frac=.9)

y = df['packet_label'].to_frame()
x = df.drop('packet_label', axis=1)
print(x)
print(y)

print('---- output sets ----')
print('x_'+dataset+':')
print(x)
print(x.info())
print('y_'+dataset+':')
print(y)
print(y.info())
sys.stdout.flush()

print('=========== saving data ============')
print('saving', direction, dataset, 'data to:')
x_path = SAVE_DIR + direction + '_x_' + dataset + '.parq'
y_path = SAVE_DIR + direction + '_y_' + dataset + '.parq'
print('x:', x_path)
print('y:', y_path)
sys.stdout.flush()

# write data to disk
print('writing x_'+dataset+'...')
sys.stdout.flush()
x.to_parquet(x_path)
print('...done.')
sys.stdout.flush()
print('writing y'+dataset+'...')
sys.stdout.flush()
y.to_parquet(y_path)
print('...done.')
sys.stdout.flush()

end_time = datetime.now()
total_time = end_time - start_time

# print runtime information
print('data split complete.')
print('Start time:', start_time)
print('End time:', end_time)
print('Elapsed:', total_time)
sys.stdout.flush()