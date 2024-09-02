"""
=============================================================================
data_processing.py
A. Huang
Updated 5/9/2023
=============================================================================
Data exploration and data cleaning/munging to prepare for model training
=============================================================================
Objectives:
1. Read from Parquet files and convert to DataFrames.
2. Convert or drop 12 different labels into 4 classes.
3. Convert IP features into ASNs and one-hot encode ASNs.
4. Correlate IP features to latitude and longitude and scale values.
5. Convert ports into TCP/UDP feature and one-hot encoded categories.
6. Scale features.
7. Convert DataFrames into numpy arrays and output into .npy binaries for 
   further use.
=============================================================================
"""
################
# ACTION ITEMS #
################

import numpy as np
import pandas as pd
import geoip2.database
import geoip2.errors
import sys
#import concurrent.futures
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from socket import inet_ntoa
from struct import pack
from maxminddb import Reader
from datetime import datetime
from gc import collect
from joblib import dump, load
from os import makedirs
from fastparquet import write
from tqdm import tqdm

#############
# GLOBALS   #
#############

## Path to GeoIP2 database file
GEOIP2_DB_PATH = '../geoip2db/GeoLite2-ASN.mmdb'
GEOIP2_DB_CITY_PATH = '../geoip2db/GeoLite2-City.mmdb'

DATA_DIR = '/data/alexander.huang/data/1013_data/'
# True to scale and norm, False for scaling only
# using sklearn StandardScaler and sklearn Normalizer in Pipeline
# StandardScaler will scale by feature (column-wise), Normalizer will scale by sample (row-wise)
NORM = True
## Label values from Table IV in NETCOM paper
# Combined allowed traffic with dropped traffic, remove interior and exterior only traffic.
# Implemented in split_dataset() below.
# Inside = inside tap; outside = outside tap
#   Model 1:
#       1. Inbound allowed
#       2. Inbound blocked 
#   Model 2:
#       3. Outbound allowed
#       4. Outbound blocked 
# Update: Use only classes 1, 2, 4, and 8; drop all others

#########################
# Function Definitions  #
#########################
def asn_lookup(input_ip_list):
    """
    Takes input list of 32-bit IP addresses in base 10 integer format and outputs
    list of corresponding autonomous system numbersj.

    Using Maxmind GeoIP2 database:
        https://pypi.org/project/geoip2/ 
        https://www.maxmind.com/en/geoip2-databases
        https://github.com/maxmind/GeoIP2-python#database-usage 

    TODO: Using just observed ASNs for now, future work to use all possible
          ASNs - 65535 possible ASNs

    Parameters:
        <list> input_ip_list:
            Input list of 32-bit integers representing IPv4 addresses.

    Output:
        <list> out_list:
            Output list of corresponding autonomous system numbers based
            on GeoIP2 database lookup.
    """
    total = len(input_ip_list)
    count = 0
    with geoip2.database.Reader(GEOIP2_DB_PATH) as read:
        out_list = []
        for entry in input_ip_list:
            try:
                # convert 32bit integer to dotted quad string
                entry = inet_ntoa(pack('!I', entry))
                # look up IP in GeoIP2 database
                db_item = read.asn(entry)
                # add to output list

                out_list.append(db_item.autonomous_system_number)
            except(geoip2.errors.AddressNotFoundError):
                # if IP is not in database, use ASN 0
                out_list.append(0)
            count += 1
            ratio = float(count)/float(total)*100
            if(ratio%10.0 == 0.0):
                print('ASN percent complete: ' + str(round(ratio,2)) + '%')
                sys.stdout.flush()
    return out_list

def lat_long_lookup(input_ip_list):
    """
    Takes input list of 32-bit IP addresses in base 10 integer format and outputs
    list of corresponding latitudes and longitudes in two separate lists.

    Using Maxmind GeoIP2 database:
        https://pypi.org/project/geoip2/ 
        https://www.maxmind.com/en/geoip2-databases
        https://github.com/maxmind/GeoIP2-python#database-usage 

    Parameters:
        <list> input_ip_list:
            Input list of 32-bit integers representing IPv4 addresses.

    Output:
        <list> lat_list:
            Output list of corresponding latitudes from database lookup. 

        <list> long_list:
            Output list of corresponding longitudes from database lookup. 
    """
    with geoip2.database.Reader(GEOIP2_DB_CITY_PATH) as read:
        lat_list = []
        long_list = []
        lat_counter = 0
        long_counter = 0
        for entry in tqdm(input_ip_list):
            try:
                # convert 32bit integer to dotted quad string
                entry = inet_ntoa(pack('!I', entry))
                # look up IP in GeoIP2 database
                db_item = read.city(entry)
                # add to output list
                if db_item.location.latitude == None:
                    lat_counter+=1
                    lat_list.append(0.0)
                else:
                    lat_list.append(db_item.location.latitude)

                if db_item.location.longitude == None:
                    long_counter+=1
                    long_list.append(0.0)
                else:
                    long_list.append(db_item.location.longitude)

            except(geoip2.errors.AddressNotFoundError):
                # if IP is not in database, use coord 0
                lat_list.append(0.0) 
                long_list.append(0.0)

    print('None objs found in GEO lookup:')
    print('latitude:', lat_counter)
    print('longitude:', long_counter)
    return lat_list, long_list

def cc_lookup(input_ip_list):
    """
    Takes input list of 32-bit IP addresses in base 10 integer format and outputs
    list of corresponding autonomous system numbersj.

    Using Maxmind GeoIP2 database:
        https://pypi.org/project/geoip2/ 
        https://www.maxmind.com/en/geoip2-databases
        https://github.com/maxmind/GeoIP2-python#database-usage 

    TODO: Using just observed ASNs for now, future work to use all possible
          ASNs - 65535 possible ASNs

    Parameters:
        <list> input_ip_list:
            Input list of 32-bit integers representing IPv4 addresses.

    Output:
        <list> out_list:
            Output list of corresponding autonomous system numbers based
            on GeoIP2 database lookup.
    """
    counter = 0
    with geoip2.database.Reader(GEOIP2_DB_CITY_PATH) as read:
        out_list = []
        for entry in tqdm(input_ip_list):
            try:
                entry = inet_ntoa(pack('!I', entry))
                # look up IP in GeoIP2 database
                db_item = read.city(entry)
                # add to output list
                if db_item.country.iso_code == None:
                    counter +=1
                    out_list.append('CC_NOT_IN_DB')
                else:
                    out_list.append(db_item.country.iso_code)
            except(geoip2.errors.AddressNotFoundError):
                out_list.append('CC_NOT_IN_DB')
    print('None objs found in cc lookup:', counter)
    return out_list


#########
# Main  #
#########
if __name__ == '__main__':
    # track runtime (not including imports)
    start_time = datetime.now()
    # inbound or outbound
    direction = sys.argv[1]
    # train or test/val
    dataset = sys.argv[2]

    file = DATA_DIR + direction+'_read_'+dataset+'.parq' 
    print('======================', direction, dataset, 'data processing ======================')
    print('reading file:', file)
    df = pd.read_parquet(file)
    #if dataset == 'test':
    #n = 9000000
    #df = df.sample(n)
    #print(direction, dataset, 'dataset count:', n)
    sys.stdout.flush()


    if(direction == 'inbound'):
        # drop unused features
        df = df.drop(['inside_timestamp',
                      'outside_timestamp',
                     'inside_ip_src',
                     'inside_ip_dst',
                     'inside_tcp_win_size',
                     'inside_tcp_flags',
                     'inside_ip_len',
                     'inside_ip_ttl',
                     'inside_port_src',
                     'inside_port_dst',
                     'inside_ip_protocol',
                     'inside_ip_ttl',
                     'outside_tcp_flags',
                     'outside_ip_dst'], axis=1)

        # identify remaining features used in loops later
        ip_features = ['outside_ip_src']

        numerical_features = ['outside_ip_len',
                              'outside_ip_src_LAT',
                              'outside_ip_src_LONG',
                              'outside_ip_ttl',
                              'outside_tcp_win_size']

        categorical_features = ['outside_ip_protocol',
                                'outside_ip_src',
                                'outside_port_src',
                                'outside_port_dst',
                                ]

    elif(direction == 'outbound'):
        df = df.drop(['outside_timestamp',
                      'inside_timestamp',
                     'outside_ip_src', 
                     'outside_ip_dst',
                     'outside_tcp_win_size',
                     'outside_tcp_flags',
                     'outside_ip_len',
                     'outside_ip_ttl',
                     'outside_port_src',
                     'outside_port_dst',
                     'outside_ip_protocol',
                     'outside_ip_ttl',
                     'inside_tcp_flags',
                     'inside_ip_src'], axis=1)

        ip_features = ['inside_ip_dst']

        numerical_features = ['inside_ip_len',
                              'inside_ip_dst_LAT',
                              'inside_ip_dst_LONG',
                              'inside_ip_ttl',
                              'inside_tcp_win_size']
        
        categorical_features = ['inside_ip_protocol',
                                'inside_ip_dst',
                                'inside_port_src',
                                'inside_port_dst']

    print('dataframe info for ', direction, dataset, 'data')
    df.info()
    sys.stdout.flush()

    # lat_long lookups
    print(direction, 'lat long lookup...')
    sys.stdout.flush()
    for feature in ip_features:
        print('feature:', feature)
        df[feature+'_LAT'], df[feature+'_LONG'] = lat_long_lookup(df[feature].tolist())
    print('complete.')
    sys.stdout.flush()

    # country code lookups
    print(direction, 'cc lookup...')
    sys.stdout.flush()
    for feature in ip_features:
        print('feature:', feature)
        df[feature] = cc_lookup(df[feature].tolist())
    print('complete.')
    sys.stdout.flush()

    ''' asn lookups (unused, generates too many features, feature matrix too sparse)
    #print(direction, 'asn lookup...')
    #sys.stdout.flush()
    #for feature in ip_features:
    #    print('feature:', feature)
    #    df[feature] = asn_lookup(df[feature].tolist())
    #print('complete.')
    #sys.stdout.flush()
    '''
    
    if(direction == 'inbound'):
        # replace high ports (>1023) with -1
        df.loc[df.outside_port_src >= 1024, 'outside_port_src'] = -1
        df.loc[df.outside_port_dst >= 1024, 'outside_port_dst'] = -1 

    elif(direction == 'outbound'):
        # replace high ports (>1023) with -1
        df.loc[df.inside_port_src >= 1024, 'inside_port_src'] = -1 
        df.loc[df.inside_port_dst >= 1024, 'inside_port_dst'] = -1 

    df.to_parquet(DATA_DIR + direction+'_dblookup_' + dataset + '.parq')
    # FIXME: read from parquet if resuming from previous run (to save time)
    #df = pd.read_parquet('outbound_dblookup_test_.parq')

    print('\ndatabase lookup results') 
    if(direction == 'inbound'):
        print('\ninbound_df:')
    elif(direction == 'outbound'):
        print('\noutbound_df:')
    print(df.info())
    print(df.head())
    sys.stdout.flush()

    print('------ categorical encoding -------')

    if dataset=='train':
        cat_dict = {}
        # one hot encoding for categorical features using pandas
        for feature in categorical_features:
            # set cf to a pd.Series of all values in one feature column
            cf = df[feature]
            '''
            use get_dummies to generate one hot array from feature column values
            column names will be "<feature>_<value>" i.e. outside_ip_src_US for
            feature "outside_ip_src" and value "US"
            '''
            cf = pd.get_dummies(cf.astype(str),
                                prefix=feature, dtype=np.int8)
            # save list of resulting features for reproduction when generating test set 
            cat_dict[feature] = cf.columns.tolist()

            '''
            unseen features were removed for final thesis -- unseen counts were low;
            samples unseen were just dropped
            '''
            # init unseen category to 0
            #cf[feature+'_unseen'] = np.int8(0)

            # order columns into reproducible order for test set later
            cf = cf.sort_index(axis=1)
            print(cf.info())
            df = pd.concat([df, cf], axis=1)
            df = df.drop(feature, axis=1)
            del cf
            collect()

        # sklearn StandardScaler will normalize by column (per feature)
        scaler = ('scaler', ColumnTransformer([('in_num_scale', 
                                                StandardScaler(),
                                                numerical_features)],
                                                remainder='passthrough',
                                                verbose_feature_names_out=False)) 

        # sklearn Normalizer will normalize by row (per sample)
        norm = ('norm', ColumnTransformer([('in_num_norm', 
                                            Normalizer(),
                                            numerical_features)],
                                            remainder='passthrough',
                                            verbose_feature_names_out=False)) 
        # scale and normalize
        if NORM == True:
            pipe = Pipeline(steps=[scaler, norm]).set_output(transform='pandas')
        # just scale
        elif NORM == False:
            pipe = Pipeline(steps=[scaler]).set_output(transform='pandas')
        else:
            raise ValueError('Norm variable not set. Recieved:', NORM)

        # save list of one_hot encoded features
        # dictionary is key:<feature> : value:<one-hot names>
        dump(cat_dict, DATA_DIR + direction+'_cat.dict')
        print('\n', direction, 'fit/transform...')
        print(df.info())
        sys.stdout.flush()
        enc = pipe.fit(df)
        # save encoder for generating test set with same scaling/normalization
        # and for reversibility of feature matrix X back to original values
        dump(enc, DATA_DIR + direction+'_pipe.enc')

    # process test/validation set, must be same features as training set
    elif dataset == 'test' or dataset == 'val':
        # load saved feature lists and encoder from training set
        cat_dict = load(DATA_DIR + direction+'_cat.dict')
        enc = load(DATA_DIR + direction+'_pipe.enc')
            
        for feature in categorical_features:
            cf = df[feature]
            print('cat_dict feature list:', cat_dict[feature])
            cat_list = []
            # creates list of features from loaded category dictionary without prefix
            # will check values by string comparison later to determine if feature is unseen
            # in training set
            for x in cat_dict[feature]:
                y = x[len(feature+'_'):]
                cat_list.append(y)
            print('cat_list with removed prefix:', cat_list)

            # identify unseen feature values and replace with "unseen"
            cf = cf.apply(lambda x: 'unseen' if str(x) not in cat_list else str(x))

            # drop data sample if it contains an unseen feature
            # FIXME: untested, should use column transformer (below) instead
            cf.drop(cf.loc[cf[feature]=='unseen'].index, inplace=True)

            # if no unseen values in feature, make empty unseen column to match training data
            cf = pd.get_dummies(cf.astype(str),
                                prefix=feature, dtype=np.int8)
            print('get dummies cols:', cf.columns)

            ''' unneeded if dropping unseen features
            cf[feature+'_unseen'] = cf.get(feature+'_unseen', np.int8(0))

            # make empty columns for keys not seen in test set but seen in training set
            for key in cat_dict[feature]:
                if key not in cf.columns:
                    print('key:', key)
                    cf[key] = np.int8(0)
            ''' 

            # reorder all columns to same order as training set
            cf = cf.sort_index(axis=1)
            print(cf.info())
            df = pd.concat([df, cf], axis=1)
            df = df.drop(feature, axis=1)
            del cf
            collect()

    # transform numerical features
    processed_df = enc.transform(df)
    del df
    collect() 

    # FIXME: replaced in thesis runs with pandas.get_dummies code above
    # if dropping unseen features, sklearn column transformers should be more efficient
    # and easier to manage

    '''data pipe w/ sklearn pipelines
    if(dataset == 'train'):
        print('setting up', direction, 'data pipeline')

        labels = ['packet_label']

        if(direction == 'inbound'):
            inbound_numerical_features = ['outside_timestamp',
                                        'outside_ip_len',
                                        'outside_ip_src_LAT',
                                        'outside_ip_src_LONG',
                                        'outside_ip_ttl',
                                        'outside_tcp_win_size']

            inbound_categorical_features = ['outside_ip_protocol',
                                            'outside_ip_src',
                                            'outside_port_src',
                                            'outside_port_dst',
                                            'outside_tcp_flags']

            inbound_scaler = ('scaler', ColumnTransformer([('in_num', 
                                                            StandardScaler(),
                                                            inbound_numerical_features)],
                                                            remainder='passthrough',
                                                            verbose_feature_names_out=False)) 
            
            inbound_encoder = ('encoder', ColumnTransformer([('in_cat',
                                                            OneHotEncoder(sparse_output=False,
                                                                            dtype=np.uint8,
                                                                            handle_unknown='error'),
                                                            inbound_categorical_features)],
                                                            remainder='passthrough',
                                                            verbose_feature_names_out=False))

            inbound_pipe = Pipeline(steps=[inbound_scaler, inbound_encoder]).set_output(transform='pandas')

            print('\n', direction, ' fit/transform...')
            print(df.info())
            sys.stdout.flush()
            #processed_df = inbound_pipe.fit_transform(df)
            enc = inbound_pipe.fit(df)
            dump(enc, 'inbound_pipe.enc')
            processed_df = enc.transform(df)
        
        elif(direction == 'outbound'):
            outbound_numerical_features = ['inside_timestamp',
                                        'inside_ip_len',
                                        'inside_ip_dst_LAT',
                                        'inside_ip_dst_LONG',
                                        'inside_ip_ttl',
                                        'inside_tcp_win_size']
            
            outbound_categorical_features = ['inside_ip_protocol',
                                            'inside_ip_dst',
                                            'inside_port_src',
                                            'inside_port_dst',
                                            'inside_tcp_flags']

            outbound_scaler = ('scaler', ColumnTransformer([('out_num', 
                                                            StandardScaler(),
                                                            outbound_numerical_features)],
                                                            remainder='passthrough',
                                                            verbose_feature_names_out=False)) 
            
            outbound_encoder = ('encoder', ColumnTransformer([('out_cat',
                                                            OneHotEncoder(sparse_output=False,
                                                                            dtype=np.uint8,
                                                                            handle_unknown='error'),
                                                            outbound_categorical_features)],
                                                            remainder='passthrough',
                                                            verbose_feature_names_out=False))
            # outbound data processing
            outbound_pipe = Pipeline(steps=[outbound_scaler, outbound_encoder]).set_output(transform='pandas')

            print('\n', direction, 'fit/transform...')
            print(df.info())
            sys.stdout.flush()
            enc = outbound_pipe.fit(df)
            dump(enc, 'outbound_pipe.enc')
            processed_df = enc.transform(df)

    elif(dataset == 'test'):
        if direction == 'outbound':
            # if test set, load previously fitted encoder from the training set
            fitted_pipe = load('outbound_pipe.enc')
            known_categories = fitted_pipe.categories_


            processed_df = fitted_pipe.transform(df)
        elif direction == 'inbound':
            # if test set, load previously fitted encoder from the training set
            fitted_pipe = load('inbound_pipe.enc')

            processed_df = fitted_pipe.transform(df)
        else:
            raise ValueError('Training encoder not found to fit test data')

    print('\nresult of', direction, 'fit_transform')
    print(processed_df.info())
    '''

    # renames >1023 port feature into HIGH from -1 for readability
    processed_df.rename(columns={'inside_port_src_-1':'inside_port_src_HIGH',
                                 'inside_port_dst_-1':'inside_port_dst_HIGH'},
                                 inplace=True)

    processed_df.rename(columns={'outside_port_src_-1':'outside_port_src_HIGH',
                                 'outside_port_dst_-1':'outside_port_dst_HIGH'},
                                 inplace=True)
    
    #processed_df = pd.read_parquet('in_fit.parq')

    print('\n----------------- RESULTS ------------------\n')
    print('-------------', direction, dataset, ' result information-------------')
    # print info on inbound data
    processed_df.info()
    print(processed_df.head(3))
    print('\noutput label value counts (', direction, dataset, 'data):')
    print(processed_df['packet_label'].value_counts())
    sys.stdout.flush()

    print('\nwriting', direction, dataset, 'dataframe to parquet...')
    sys.stdout.flush()
    outfile = DATA_DIR + direction + '_proc_' + dataset+'.parq'
    processed_df.to_parquet(outfile)
    del processed_df 
    collect()
    print(direction, ' data processing complete. Path:', outfile)
    sys.stdout.flush()

    end_time = datetime.now()
    total_time = datetime.now() - start_time

    # print runtime information
    print('Start time:', start_time)
    print('End time:', end_time)
    print('Elapsed:', total_time)