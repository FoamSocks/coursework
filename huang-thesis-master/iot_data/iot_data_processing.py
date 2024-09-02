import pandas as pd
import geoip2.database
import geoip2.errors
import sys
from socket import inet_ntoa
from struct import pack
from maxminddb import Reader
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from os import listdir

LABELED_PACKET_PATH = './labeled_packets'

GEOIP2_DB_PATH = '../geoip2db/GeoLite2-ASN.mmdb'
GEOIP2_DB_CITY_PATH = '../geoip2db/GeoLite2-City.mmdb'

PREPROCESSING_FILE = 'iot_data.parq'
OUTPUT_FILE = 'processed_iot_data.parq'

def read_file(from_parquet=False):
    """ 
    Reads from Parquet file into a Pandas dataframe.

    Parameters:
        <int> fileno: 
            Index number of file within the directory list.

    Output:
        <pandas.dataframe> packet_df: 
            Dataframe of packets read from Parquet file at designated index. 
    """
    output_df = pd.DataFrame()
    if from_parquet == True:
        print('Reading parquet file from path:', PREPROCESSING_FILE)
        sys.stdout.flush()
        output_df = pd.read_parquet(PREPROCESSING_FILE)
        print('done.')
        sys.stdout.flush()

    elif from_parquet == False:
        print("Reading file from path:", LABELED_PACKET_PATH) 

        # get directory listing
        directory = listdir(LABELED_PACKET_PATH)
        num_files = len(directory)
        print('Number of files:', num_files)
        #random = np.random.randint(num_files, size=fileno)

        # build dataset out all files in the directory
        for entry in range(num_files): 
            target_path = LABELED_PACKET_PATH + '/' + directory[entry] 
            packet_df = pd.read_parquet(target_path)
            print('Read file ('+str(entry)+'): '+target_path)
            output_df = pd.concat([output_df, packet_df], axis=0, copy=False)
            sys.stdout.flush()
        output_df = output_df.reset_index(drop=True)

        print('outputing pre-processing dataframe...')
        sys.stdout.flush()
        output_df.to_parquet(PREPROCESSING_FILE)
        print('done.')
        print('')
        sys.stdout.flush()

    return output_df 

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
    return out_list

def city_lookup(input_ip_list):
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
        for entry in input_ip_list:
            try:
                # convert 32bit integer to dotted quad string
                entry = inet_ntoa(pack('!I', entry))
                # look up IP in GeoIP2 database
                db_item = read.city(entry)
                # add to output list
                lat_list.append(db_item.location.latitude)
                long_list.append(db_item.location.longitude)
            except(geoip2.errors.AddressNotFoundError):
                # if IP is not in database, use ASN 0
                lat_list.append(0.0) 
                long_list.append(0.0)
    return lat_list, long_list

def database_lookups(input_df):
    """
    Takes input dataframe and produces output dataframe with src/dst_ip 
    converted to ASNs and ASNs one-hot encoded

    Parameters:
        <pandas.dataframe> input_df:
            Input dataframe with 32-bit integers representing IPv4 addresses
            in as src_ip and dst_ip features

    Output:
        <pandas.dataframe> output_df:
            Output dataframe with one-hot encoded ASNs and src_ip/dst_ip features
            removed
    """
    output_df = pd.DataFrame()

    # target IP features to convert to ASN
    ip_src_series = input_df['ip_src']
    ip_dst_series = input_df['ip_dst']

    # for each of four features, look up matching ASN for every IP
    print('')
    print('asn lookup...')
    sys.stdout.flush()
    output_df['ip_src_ASN'] = asn_lookup(ip_src_series.tolist())
    output_df['ip_dst_ASN'] = asn_lookup(ip_dst_series.tolist())
    print('...complete.')
    sys.stdout.flush()

    print('')
    print('ASN processing...get_dummies')
    sys.stdout.flush()
    # make one-hot encoded features out of ASN lookups from GeoIP2 database
    output_df = pd.get_dummies(output_df.astype(str), 
                               prefix=['ip_src_asn', 
                                       'ip_dst_asn',])

    print('...complete.')
    sys.stdout.flush()

    print('')
    print('lat/long lookup...')
    sys.stdout.flush()

    input_df['ip_src_LAT'], input_df['ip_src_LONG'] = city_lookup(ip_src_series.tolist())
    input_df['ip_dst_LAT'], input_df['ip_dst_LONG'] = city_lookup(ip_dst_series.tolist())

    print('...complete.')
    sys.stdout.flush()

    # drop IP features, to be replaced by one-hot ASN features
    input_df = input_df.drop(['ip_src',
                              'ip_dst'], 
                              axis=1)

    # reset indices to marry columns between input and output dataframe
    input_df = input_df.reset_index(drop=True)
    output_df = output_df.reset_index(drop=True)

    # concatenate input_df with other features, and output_df with new ASN features
    output_df = pd.concat([input_df, output_df], axis=1, copy=False)
    sys.stdout.flush()

    return output_df

def process_ports(input_df):
    """
    Takes input dataframe and produces one-hot ecoding for each of the 
    ports if < 1024 and in a single 'HIGH' feature for each of 4 inside/outside
    src/dst combinations. 

    Parameters:
        <pandas.dataframe> input_df:
            Input dataframe with 19 features containing unprocessed port data

    Output:
        <pandas.dataframe> output_df:
            Output dataframe with additional features after processing
    """
    # Port Features: outside_port_src, outside_port_dst, inside_port_src, inside_port_dst

    output_df = input_df

    # Print values for IP protocol feature
    print('\nvalues for ip_protocol feature')
    print(output_df['ip_protocol'].value_counts())
    sys.stdout.flush()

    print('\nIP protocol to binary is_tcp feature....')
    sys.stdout.flush()

    # transform IP protocol 6 and 17 to binary 'is_tcp' feature
    # replace 6 with 1, 17 with 0
    output_df.replace({'ip_protocol': {6:1, 17:0}})
    output_df.rename(columns={'ip_protocol':'is_tcp'})

    print('...complete.')
    sys.stdout.flush()

    # print values for port features
    print('\nvalue counts for IP port features')
    print(output_df['port_src'].value_counts())
    print(output_df['port_dst'].value_counts())
    sys.stdout.flush()

    # select port features
    working_df = output_df.loc[:, ['port_src',
                                   'port_dst']]

    # drop old port features
    output_df = output_df.drop(['port_src',
                                'port_dst'], axis=1)

    print('\nworking df contents')
    print(working_df.head())
    print(working_df.info())
    sys.stdout.flush()

    # temporarily assign -1 for high port
    # if port < 1024, keep value, otherwise replace with -1
    print('\nreplacing > 1023 with -1')
    sys.stdout.flush()
    working_df.loc[working_df.port_src >= 1024, 'port_src'] = -1 
    working_df.loc[working_df.port_dst >= 1024, 'port_dst'] = -1
    print('\nresults:') 
    print(working_df['port_src'].value_counts())
    print(working_df['port_dst'].value_counts())
    sys.stdout.flush()

    print('\nport one-hot processing...get_dummies')
    sys.stdout.flush()

    # one-hot encode port features
    working_df = pd.get_dummies(working_df.astype(str), 
                                prefix=['port_src',
                                        'port_dst'])


    # rename high port column (-1) to more descriptive name
    working_df = working_df.rename(columns={'port_src_-1':'port_src_HIGH',
                                            'port_dst_-1':'port_dst_HIGH'})

    # concatenate new encoded port features to output
    output_df = pd.concat([output_df, working_df], axis=1, copy=False)
    sys.stdout.flush()

    print('...complete.')
    sys.stdout.flush()
    
    return output_df

def process_flags(input_df):
    output_df = pd.DataFrame()
    output_df = input_df

    flags_df = output_df.loc[:, ['tcp_flags']]
    output_df = output_df.drop(['tcp_flags'], axis=1)

    print('\nprocessing tcp flags...get_dummies')
    sys.stdout.flush()
    flags_df = pd.get_dummies(flags_df.astype(str), 
                              prefix=['tcp_flags'])
    
    output_df = pd.concat([output_df, flags_df], axis=1, copy=False)

    print('...done')
    sys.stdout.flush()

    return output_df


if __name__ == '__main__':
    start_time = datetime.now()

    # FIXME change to read from file
    #df = read_file()
    df = pd.read_parquet(PREPROCESSING_FILE)

    # FIXME: use sample or groupby sample
    df = df.sample(frac=0.01)
    sys.stdout.flush()

    with pd.option_context(
        'display.max_rows', None,
        'display.max_columns', None,
        'display.precision', 4
    ):
        print(df.head(5))
    
    print('-------------- Dataset Information -------------')
    print(df.info())
    sys.stdout.flush()

    output_df = pd.DataFrame()
    print('calling process ports')
    output_df = process_ports(df)
    sys.stdout.flush()

    # FIXME: TCP Flags feature doesn't make sense, drop for now.
    '''
    print('calling process flags')
    output_df = process_flags(output_df) 
    sys.stdout.flush()
    '''
    output_df.drop(['tcp_flags'], axis=1)   # FIXME Delete after fixing tcp_flags feature

    print('calling database lookups')
    output_df = database_lookups(output_df)
    sys.stdout.flush()

    output_df.reset_index(drop=True, inplace=True)

    numerical_features = ['timestamp, ip_len, ip_ttl']

    print('Scaling numerical features....')
    sys.stdout.flush()
    # scale numerical features with standard scaler
    
    scaler = StandardScaler()
    output_df[numerical_features] = scaler.fit_transform(output_df[numerical_features])
    print('complete.')
    sys.stdout.flush()

    print('-------------- Output Dataframe Information -------------')
    output_df.info()
    with pd.option_context(
        'display.max_rows', None,
        'display.max_columns', None,
        'display.precision', 4
    ):
        print(output_df.head(3))
    sys.stdout.flush()

    print('Exporting to parquet:', OUTPUT_FILE)
    sys.stdout.flush()

    output_df.to_parquet(OUTPUT_FILE)

    print('done.')
    sys.stdout.flush()

    # runtime information
    end_time = datetime.now()
    total_time = end_time - start_time

    print('Start time:', start_time)
    print('End time:', end_time)
    print('Elapsed:', total_time)