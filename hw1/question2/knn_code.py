import numpy.random
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt


def read_data(data_dir='train'):
    """
    Read the corresponding directory based on the 'data_dir', merge the fake data with label '1' and real data with
    label '0' into a list, then apply numpy to shuffle if the data is training data. In the end, return the data and
    corresponding labels in two list.
    @param data_dir: the directory of the dataset, should be either 'dev' or 'train'
    @return: the dataset and corresponding labels in two seperate lists
    """
    data_with_label = []
    # load the fake data and make label values '1'
    with open(f'data/{data_dir}/clean_fake_{data_dir}.txt') as file:
        lines = file.readlines()
        for line in lines:
            data_with_label.append({"text": line.rstrip(), "label": 1})

    # keep track of the number of fake data samples in 'data_dir' dataset
    fake_data_number = len(data_with_label)
    print(f'{data_dir} dataset contains {fake_data_number} fake data.')

    # load the real data and make label values '0'
    with open(f'data/{data_dir}/clean_real_{data_dir}.txt') as file:
        lines = file.readlines()
        for line in lines:
            data_with_label.append({"text": line.rstrip(), "label": 0})

    # keep track of the number of real data samples in 'data_dir' dataset
    real_data_number = len(data_with_label)-fake_data_number
    print(f'{data_dir} dataset contains {real_data_number} real data.')

    df = pd.DataFrame(data_with_label)
    # If it is training data, should be shuffled for a better performance when classifying
    if data_dir == 'train':
        df = df.sample(frac=1).reset_index(drop=True)

    return df['text'].values, df['label'].values


def load_data(train_data_value):
    """
    Use CountVectorizer to fit the training data values passed as parameter.
    @param train_data_value: training data values.
    @return: the CountVectorizer that already fit training data.
    """
    vectorizer = CountVectorizer()
    vectorizer.fit(train_data_value)
    return vectorizer


def select_knn_model(dist_type='euclidean'):
    """
    Load the training data and validation data, use the vectorizer returned by load_data function to transform the
    training data and validation data. For all the neighbors range from 1 to 20, apply the corresponding
    KNeighborsClassifier to fit the training data, in the end, score the training data and validation data to plot the
    accuracy.
    @param dist_type: type of metric used in KNeighborsClassifier
    """
    train_data_value, train_data_label = read_data('train')
    dev_data_value, dev_data_label = read_data('dev')
    vecotrizer = load_data(train_data_value)

    # vecotrizer transform the dev data to vectors
    x_dev = vecotrizer.transform(dev_data_value)
    y_dev = dev_data_label
    # vecotrizer transform the training data to vectors
    x_train = vecotrizer.transform(train_data_value)
    y_train = train_data_label

    neighbors = numpy.arange(1, 21)

    train_accuracy = []
    dev_accuracy = []

    # For k from 1(inclusive) to 20(inclusive), calculate the score on both the training dataset and dev dataset,
    # put into a list for further plot.
    for _, i in enumerate(neighbors):

        neigh = KNeighborsClassifier(n_neighbors=i, metric=dist_type)
        # train the training data
        neigh.fit(x_train, y_train)

        # score the training data and validation data
        train_score = neigh.score(x_train, y_train)
        train_accuracy.append(train_score)

        dev_score = neigh.score(x_dev, y_dev)
        dev_accuracy.append(dev_score)

    # plot the accuracy
    plt.title(f'k-NN: Varying Number of Neighbors({dist_type})')
    plt.plot(neighbors, train_accuracy, label='Training Accuracy')
    plt.plot(neighbors, dev_accuracy, label='Validation Accuracy')
    plt.legend()
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Accuracy')
    plt.show()


select_knn_model()
# select_knn_model('cosine')
