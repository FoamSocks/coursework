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

import pandas as pd
import sys
from os import makedirs
import random
from datetime import datetime
from os import listdir
from gc import collect
from fastparquet import write

#############
# GLOBALS   #
#############

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

# inbound labels
LABEL_INCOMING = 1
LABEL_OUTSIDE_INBOUND_BLOCKED = 8

# outbound labels
LABEL_OUTGOING = 2
LABEL_INSIDE_OUTBOUND_BLOCKED = 4

MINUTES = 5 

DATA_DIR = '/data/alexander.huang/data/0903_data/'


#########################
# Function Definitions  #
#########################
def split_dataset(input_df):
    """
    Takes input dataframe and splits into two datasets, one with only inbound labels
    and another with only outbound labels.

    Parameters:
        <pandas.dataframe> input_df:
            Input dataframe to be split.

    Output:
        <pandas.dataframe> inbound_df:
            Dataframe containing only data labeled with inbound labels
        <pandas.dataframe> outbound_df:
            Dataframe containing only data labeled with outbound labels
    """
    inbound_df = pd.DataFrame() 
    outbound_df = pd.DataFrame() 

    print('')
    print('\nsplit_dataset() function call')
    print('input label value counts:')
    print(input_df['packet_label'].value_counts())

    inbound_df = input_df[(input_df.packet_label == LABEL_INCOMING) | (input_df.packet_label == LABEL_OUTSIDE_INBOUND_BLOCKED)]
    inbound_df = inbound_df.replace({LABEL_INCOMING:0, LABEL_OUTSIDE_INBOUND_BLOCKED:1})
    outbound_df= input_df[(input_df.packet_label == LABEL_OUTGOING) | (input_df.packet_label == LABEL_INSIDE_OUTBOUND_BLOCKED)]
    outbound_df = outbound_df.replace({LABEL_OUTGOING:0, LABEL_INSIDE_OUTBOUND_BLOCKED:1})

    print('\noutput label value counts (inbound data):')
    print(inbound_df['packet_label'].value_counts())
    print('\noutput label value counts (outbound data):')
    print(outbound_df['packet_label'].value_counts())
    sys.stdout.flush()
    del input_df
    collect()

    return inbound_df, outbound_df

def read_file(month=12, set='train'):
    """ 
    Reads from Parquet file into a Pandas dataframe.

    Parameters:
        <int> date:
            Numerical date in December 2022 to process, between 4 and 12
        <int> fileno:
            Number of parquet files to process, 0 for all

    Output:
        <pandas.dataframe> packet_df: 
            Dataframe of packets read from Parquet file at designated index. 
    """
    ## Path to labeled packet data in Parquet format on Hamming
    if(month == 12):
        LABELED_PACKET_PATH = '/NETCOM/data/bdallen/2212_new/labeled_packets'
        timestamp = 't_2022-12-'
    elif(month == 9):
        LABELED_PACKET_PATH = '/NETCOM/data/bdallen/2209_new/labeled_packets'
        timestamp = 't_2022-09-'

    # read file from 12/22 labeled packet data
    print("Reading files from path:", LABELED_PACKET_PATH) 

    # get directory listing
    directory = listdir(LABELED_PACKET_PATH)
    num_files = len(directory)
    print('Number of files:', num_files)
    #random = np.random.randint(num_files, size=fileno)
    output_df = pd.DataFrame()

    # build dataset out of selected files in the directory
    # for 2209, all parquet files are from the same day within 18 hour window
    # for 2212, packets collected over a week long period, from 12/04 1556 to 12/12 1801

    # example filename w/ timestamp: t_2022-12-04_16_01_44.390162.parq
    # number of files to read

    if set=='train':
        start_day = 5
        end_day = 9
        start_hour = 9
        end_hour = 15
        start = 0
        end = 0
        for day in range(start_day, end_day+1):
            for hour in range(start_hour, end_hour+1):
                for min in range(60):
                    train_timestamp = timestamp
                    print('reading date: '+ str(month)+'/'+str(day))
                    if day > 9:
                        train_timestamp += str(day) + '_'
                    else:
                        train_timestamp += '0' + str(day) + '_'
                    if hour > 9:
                        train_timestamp += str(hour) + '_'
                    else:
                        train_timestamp += '0' + str(hour) + '_'
                    if min > 9:
                        train_timestamp += str(min)
                    else:
                        train_timestamp += '0' + str(min)

                    print('Target timestamp (train):', train_timestamp)

                    min_df = pd.DataFrame()
                    in_min_df = pd.DataFrame()
                    out_min_df = pd.DataFrame()
                    for entry in directory:
                        if entry.startswith(train_timestamp):
                            directory.remove(entry)
                            read_path = LABELED_PACKET_PATH + '/' + entry 
                            print('Training read file:', read_path)
                            packet_df = pd.read_parquet(read_path)
                            in_packet_df = packet_df[(packet_df.packet_label == LABEL_INCOMING) | (packet_df.packet_label == LABEL_OUTSIDE_INBOUND_BLOCKED)]  
                            out_packet_df = packet_df[(packet_df.packet_label == LABEL_OUTGOING) | (packet_df.packet_label == LABEL_INSIDE_OUTBOUND_BLOCKED)]
                            in_min_df = pd.concat([in_min_df, in_packet_df], axis=0, copy=False)
                            out_min_df = pd.concat([out_min_df, out_packet_df], axis=0, copy=False)
                            del packet_df
                            collect()
                            sys.stdout.flush()
                    # shuffle samples
                    in_min_df = in_min_df.sample(n=6500)
                    out_min_df = out_min_df.sample(n=6500)
                    min_df = pd.concat([in_min_df, out_min_df], axis=0, copy=False)
                    filename = DATA_DIR + 'raw_data_train.parq'
                    end = start+len(min_df)
                    print('indexes of current dataframe:')
                    print('start:', start)
                    print('end:', end)
                    min_df.index = list(range(start, end))
                    start += len(min_df)
                    print('append output train data to:', filename)
                    try:
                        write(filename, min_df, append=True)
                    except FileNotFoundError:
                        write(filename, min_df)
                    print('done')
                    del in_min_df
                    del out_min_df
                    del min_df
                    collect()
        # validation data is 10% of training
        filename = DATA_DIR + 'raw_data_train.parq'
        output_df = pd.read_parquet(filename)
        val_df = output_df.sample(frac=0.1)
        # drop validation set indices from training set
        output_df = output_df.drop(val_df.index)

        #output_df = output_df.reset_index(drop=True)
        #filename = DATA_DIR + 'raw_data_train.parq'
        #print('output train data to:', filename)
        #output_df.to_parquet(filename)
        sys.stdout.flush()

        #val_df = val_df.reset_index(drop=True)
        filename = DATA_DIR + 'raw_data_val.parq'
        print('output val data to:', filename)
        val_df.to_parquet(filename)
        sys.stdout.flush()
        return output_df, val_df
    

            # make validation dataset from 12/11 data, different data reads from training set
            #print('data validation between train and val (should display false):')
            #print(train_datalist == val_datalist)

    elif set=='test':
        hour = 10
        start = 0
        end = 0
        for min in range(0, 60):
            test_timestamp = timestamp 
            day = 12 
            print('reading date: ' + str(month)+'/'+str(day))
            if day > 9:
                test_timestamp += str(day) +'_'
            else:
                test_timestamp += '0' + str(day) +'_'
            if hour > 9:
                test_timestamp += str(hour) + '_'
            else:
                test_timestamp += '0' + str(hour) + '_'
            if min > 9:
                test_timestamp += str(min)
            else:
                test_timestamp += '0' + str(min)

            print('Target timestamp (test):', test_timestamp)
            min_df = pd.DataFrame()
            in_min_df = pd.DataFrame()
            out_min_df = pd.DataFrame()
            for entry in directory:
                if entry.startswith(test_timestamp):
                    directory.remove(entry)
                    read_path = LABELED_PACKET_PATH + '/' + entry 
                    print('Read file:', read_path)
                    packet_df = pd.read_parquet(read_path)
                    in_packet_df = packet_df[(packet_df.packet_label == LABEL_INCOMING) | (packet_df.packet_label == LABEL_OUTSIDE_INBOUND_BLOCKED)]  
                    out_packet_df = packet_df[(packet_df.packet_label == LABEL_OUTGOING) | (packet_df.packet_label == LABEL_INSIDE_OUTBOUND_BLOCKED)]
                    in_min_df = pd.concat([in_min_df, in_packet_df], axis=0, copy=False)
                    out_min_df = pd.concat([out_min_df, out_packet_df], axis=0, copy=False)
                    del packet_df
                    collect()
                    sys.stdout.flush()

            in_min_df = in_min_df.sample(60000)
            out_min_df = out_min_df.sample(60000)
            min_df = pd.concat([in_min_df, out_min_df], axis=0, copy=False)
            filename = DATA_DIR + 'raw_data_test.parq'
            end = start+len(min_df)
            print('indexes of current dataframe:')
            print('start:', start)
            print('end:', end)
            min_df.index = list(range(start, end))
            start += len(min_df)
            print('append output train data to:', filename)
            try:
                write(filename, min_df, append=True)
            except FileNotFoundError:
                write(filename, min_df)
            print('done')
            del in_min_df
            del out_min_df
            del min_df
            collect()

        filename = DATA_DIR + 'raw_data_test.parq'
        output_df = pd.read_parquet(filename)
        sys.stdout.flush()
        return output_df
        
    else:
        raise(Exception('incorrect option for dataset'))

    '''
    if set == 'train':
        output_df = output_df.reset_index(drop=True)
        filename = 'raw_data_' + set_option + '_.parq'
        print('output train data to:', filename)
        output_df.to_parquet(filename)
        sys.stdout.flush()

        val_df = val_df.reset_index(drop=True)
        filename = DATA_DIR + 'raw_data_val_.parq'
        print('output val data to:', filename)
        val_df.to_parquet(filename)
        sys.stdout.flush()
        return output_df, val_df

    elif set =='test':
        output_df = output_df.reset_index(drop=True)
        filename = DATA_DIR + 'raw_data_' + set_option + '_.parq'
        print('output test data to:', filename)
        output_df.to_parquet(filename)
        sys.stdout.flush()
        return output_df
    '''

def read_day_file(month=12, day=5, dataset='train'):
    
    ## Path to labeled packet data in Parquet format on Hamming
    if(month == 12):
        LABELED_PACKET_PATH = '/NETCOM/data/bdallen/2212_new/labeled_packets'
        timestamp = 't_2022-12-'
    elif(month == 9):
        LABELED_PACKET_PATH = '/NETCOM/data/bdallen/2209_new/labeled_packets'
        timestamp = 't_2022-09-'

    # read file from 12/22 labeled packet data
    print("Reading files from path:", LABELED_PACKET_PATH) 

    # get directory listing
    directory = listdir(LABELED_PACKET_PATH)
    num_files = len(directory)
    print('Number of files:', num_files)
    #random = np.random.randint(num_files, size=fileno)
    output_df = pd.DataFrame()

    # example filename w/ timestamp: t_2022-12-04_16_01_44.390162.parq
    # Read all .parq files per minute, sample 6900 samples, for 24 hours
    print('reading date: '+ str(month)+'/'+str(day))
    if dataset == 'train':
        start = 0
        end = 0
        for hour in range(9, 16):
            for min in range(60):
                train_timestamp = timestamp
                if day > 9:
                    train_timestamp += str(day) + '_'
                else:
                    train_timestamp += '0' + str(day) + '_'
                if hour > 9:
                    train_timestamp += str(hour) + '_'
                else:
                    train_timestamp += '0' + str(hour) + '_'
                if min > 9:
                    train_timestamp += str(min)
                else:
                    train_timestamp += '0' + str(min)
                
                print('Target timestamp (train):', train_timestamp)

                min_df = pd.DataFrame()
                in_min_df = pd.DataFrame()
                out_min_df = pd.DataFrame()
                # build training set candidates based on timestamp
                for entry in directory:
                    if entry.startswith(train_timestamp):
                        directory.remove(entry)
                        read_path = LABELED_PACKET_PATH + '/' + entry 
                        print('Training read file:', read_path)
                        packet_df = pd.read_parquet(read_path)
                        in_packet_df = packet_df[(packet_df.packet_label == LABEL_INCOMING) | (packet_df.packet_label == LABEL_OUTSIDE_INBOUND_BLOCKED)]  
                        out_packet_df = packet_df[(packet_df.packet_label == LABEL_OUTGOING) | (packet_df.packet_label == LABEL_INSIDE_OUTBOUND_BLOCKED)]
                        in_min_df = pd.concat([in_min_df, in_packet_df], axis=0, copy=False)
                        out_min_df = pd.concat([out_min_df, out_packet_df], axis=0, copy=False)
                        del packet_df
                        collect()
                        sys.stdout.flush()
                in_min_df = in_min_df.sample(n=24000)
                out_min_df = out_min_df.sample(n=24000)
                min_df = pd.concat([in_min_df, out_min_df], axis=0, copy=False)
                filename = DATA_DIR + 'raw_data_train.parq'
                end = start+len(min_df)
                print('indexes of current dataframe:')
                print('start:', start)
                print('end:', end)
                min_df.index = list(range(start, end))
                start += len(min_df)
                print('append output train data to:', filename)
                try:
                    write(filename, min_df, append=True)
                except FileNotFoundError:
                    write(filename, min_df)
                print('done')
                del min_df
                collect()
        # validation data is 10% of training
        filename = DATA_DIR + 'raw_data_train.parq'
        output_df = pd.read_parquet(filename)
        val_df = output_df.sample(frac=0.1)
        # drop validation set indices from training set
        output_df = output_df.drop(val_df.index)

        #output_df = output_df.reset_index(drop=True)
        #filename = DATA_DIR + 'raw_data_train.parq'
        #print('output train data to:', filename)
        #output_df.to_parquet(filename)
        sys.stdout.flush()

        #val_df = val_df.reset_index(drop=True)
        filename = DATA_DIR + 'raw_data_val.parq'
        print('output val data to:', filename)
        val_df.to_parquet(filename)
        sys.stdout.flush()
        return output_df, val_df

    elif dataset=='test':
        # read 1 parq per minute for num minutes = MINUTES
        test_timestamp = timestamp 
        hour = 10 
        print('reading date: ' + str(month)+'/'+str(day))
        if day > 9:
            test_timestamp += str(day) +'_'
        else:
            test_timestamp += '0' + str(day) +'_'
        
        if hour > 9:
            test_timestamp += str(hour) + '_'
        else:
            test_timestamp += '0' + str(hour) + '_'

        print('Target timestamp (test):', test_timestamp)
        test_set = []

        for entry in directory:
            if entry.startswith(test_timestamp):
                test_set.append(entry)
        
        random.shuffle(test_set)

        #test_sample = random.sample(test_set, k=50)
        #for file in test_sample:
        start = 0
        end = 0
        for min in range(0, MINUTES):
            min_df = pd.DataFrame()
            for entry in test_set:
                if min > 9:
                    target = test_timestamp + str(min)
                else:
                    target = test_timestamp + '0' + str(min)
                if entry.startswith(target):
                    read_path = LABELED_PACKET_PATH + '/' + entry 
                    print('Read file:', entry)
                    sys.stdout.flush()
                    packet_df = pd.read_parquet(read_path)
                    packet_df = packet_df[(packet_df.packet_label == LABEL_INCOMING) | (packet_df.packet_label == LABEL_OUTSIDE_INBOUND_BLOCKED) | (packet_df.packet_label == LABEL_OUTGOING) | (packet_df.packet_label == LABEL_INSIDE_OUTBOUND_BLOCKED)]
                    min_df = pd.concat([min_df, packet_df], axis=0, copy=False)
                    del packet_df
                    collect()
                    #break
            filename = DATA_DIR + 'raw_data_test.parq'
            print('append output train data to:', filename)
            sys.stdout.flush()
            end = start+len(min_df)
            print('indexes of current dataframe:')
            print('start:', start)
            print('end:', end)
            min_df.index = list(range(start, end))
            start += len(min_df)
            try:
                write(filename, min_df, append=True)
            except FileNotFoundError:
                write(filename, min_df)
            del min_df
            collect()
            sys.stdout.flush()

        filename = DATA_DIR + 'raw_data_test.parq'
        output_df = pd.read_parquet(filename)
        sys.stdout.flush()
        return output_df

#########
# Main  #
#########
if __name__ == '__main__':
    # track runtime (not including imports)
    start_time = datetime.now()

    set_option = sys.argv[1]
    print('dataset to read:', set_option)
    print('data directory:', DATA_DIR)

    makedirs(DATA_DIR, exist_ok=True)

    # construct dataframe, reading from labeled packet parquets
    if set_option == 'train':
        df, val_df = read_file(12, set_option)
        #df , val_df = read_day_file(12, 5, set_option)
    elif set_option == 'test':
        df = read_file(12, set_option)
        #df = read_day_file(12, 6, set_option)

    sys.stdout.flush()

    # print basic dataframe info
    print('\nRead data---------------')

    with pd.option_context(             # display all dataframe columns
        'display.max_rows', None, 
        'display.max_columns', None,
        'display.precision', 4
        ):
        print(df.head(5))

    # print info, including na values to verify clean
    print('-----------------Dataset Information-----------------')
    df.info()
    sys.stdout.flush()

    
    # drop non-salient features, 
    # TCP sequence numbers have no context outside of a TCP flow
    df = df.drop(['outside_tcp_seq',
                'outside_tcp_ack',
                'inside_tcp_seq',
                'inside_tcp_ack'], axis=1)

    # split the dataset into inbound and outbound packets for separate models
    print('\ncalling split_dataset----\n')
    inbound_df, outbound_df = split_dataset(df)

    del df
    collect()

    print('\n'+set_option+' dataset label counts:')
    print('inbound df')
    print(inbound_df['packet_label'].value_counts())
    print('outbound df')
    print(outbound_df['packet_label'].value_counts())
    sys.stdout.flush()


    print('writing inbound '+set_option+' parquet...')
    sys.stdout.flush()
    inbound_df.to_parquet(DATA_DIR+'inbound_read_'+set_option+'.parq')
    print('...complete')
    sys.stdout.flush()
    
    print('writing outbound '+set_option+' parquet...')
    sys.stdout.flush()
    outbound_df.to_parquet(DATA_DIR+'outbound_read_'+set_option+'.parq')
    print('...complete')
    sys.stdout.flush()

    if set_option == 'train':
        print('\nRead val data ---------------')

        with pd.option_context(             # display all dataframe columns
            'display.max_rows', None, 
            'display.max_columns', None,
            'display.precision', 4
            ):
            print(val_df.head(5))

        # print info, including na values to verify clean
        print('-----------------Dataset Information-----------------')
        val_df.info()
        sys.stdout.flush()

        # drop non-salient features, 
        # TCP sequence numbers have no context outside of a TCP flow
        val_df = val_df.drop(['outside_tcp_seq',
                              'outside_tcp_ack',
                              'inside_tcp_seq',
                              'inside_tcp_ack'], axis=1)

        # split the dataset into inbound and outbound packets for separate models
        print('\ncalling split_dataset----\n')
        inbound_df, outbound_df = split_dataset(val_df)

        del val_df
        collect()

        print('\n val dataset label counts:')
        print('inbound df')
        print(inbound_df['packet_label'].value_counts())
        print('outbound df')
        print(outbound_df['packet_label'].value_counts())
        sys.stdout.flush()


        print('writing inbound val parquet...')
        sys.stdout.flush()
        inbound_df.to_parquet(DATA_DIR + 'inbound_read_val.parq')
        print('...complete')
        sys.stdout.flush()
        
        print('writing outbound val parquet...')
        sys.stdout.flush()
        outbound_df.to_parquet(DATA_DIR + 'outbound_read_val.parq')
        print('...complete')
        sys.stdout.flush()

    end_time = datetime.now()
    total_time = datetime.now() - start_time

    # print runtime information
    print('Start time:', start_time)
    print('End time:', end_time)
    print('Elapsed:', total_time)