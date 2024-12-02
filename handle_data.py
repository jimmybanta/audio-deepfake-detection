
import pandas as pd


def data_stats(path):
    '''
    Given a path to the data meta file, return stats about the data.
    '''
    data = pd.read_csv(path)

    print(data['label'].value_counts())

    print(data['speaker'].value_counts())



def generate_labels(data, labels={
    'spoof': 1,
    'bona-fide': 0
}):
    '''
    Given the data meta dataframe, generate numeric labels for the data.
    '''
    
    data['numeric_label'] = data['label'].map(labels)

    return data


if __name__ == '__main__':

    path = 'data_meta.csv'

    data = pd.read_csv(path)

    data_with_labels = generate_labels(data)

    data.to_csv('data_meta.csv', index=False)
