from sklearn.model_selection import train_test_split


def train_test_valid_split(labels, valid_frac=0.15, test_frac=0.15):
    valid_size = int(valid_frac * labels.shape[0])
    test_size = int(test_frac * labels.shape[0])
    train_test, valid = train_test_split(labels, test_size=valid_size)
    train, test = train_test_split(train_test, test_size=test_size)
    
    return train, valid, test
