import re
import string
from collections import Counter
import numpy as np

def preprocess_ST_message(text):
    """
    Preprocesses raw message data for analysis
    :param text: String. ST Message
    :return: List of Strings.  List of processed text tokes
    """
    # Define ST Regex Patters
    REGEX_PRICE_SIGN = re.compile(r'\$(?!\d*\.?\d+%)\d*\.?\d+|(?!\d*\.?\d+%)\d*\.?\d+\$')
    REGEX_PRICE_NOSIGN = re.compile(r'(?!\d*\.?\d+%)(?!\d*\.?\d+k)\d*\.?\d+')
    REGEX_TICKER = re.compile('\$[a-zA-Z]+')
    REGEX_USER = re.compile('\@\w+')
    REGEX_LINK = re.compile('https?:\/\/[^\s]+')
    REGEX_HTML_ENTITY = re.compile('\&\w+')
    REGEX_NON_ACSII = re.compile('[^\x00-\x7f]')
    REGEX_PUNCTUATION = re.compile('[%s]' % re.escape(string.punctuation.replace('<', '')).replace('>', ''))
    REGEX_NUMBER = re.compile(r'[-+]?[0-9]+')

    text = text.lower()

    # Replace ST "entitites" with a unique token
    text = re.sub(REGEX_TICKER, ' <TICKER> ', text)
    text = re.sub(REGEX_USER, ' <USER> ', text)
    text = re.sub(REGEX_LINK, ' <LINK> ', text)
    text = re.sub(REGEX_PRICE_SIGN, ' <PRICE> ', text)
    text = re.sub(REGEX_PRICE_NOSIGN, ' <NUMBER> ', text)
    text = re.sub(REGEX_NUMBER, ' <NUMBER> ', text)
    # Remove extraneous text data
    text = re.sub(REGEX_HTML_ENTITY, "", text)
    text = re.sub(REGEX_NON_ACSII, "", text)
    text = re.sub(REGEX_PUNCTUATION, "", text)
    # Tokenize and remove < and > that are not in special tokens
    words = " ".join(token.replace("<", "").replace(">", "")
                     if token not in ['<TICKER>', '<USER>', '<LINK>', '<PRICE>', '<NUMBER>']
                     else token
                     for token
                     in text.split())

    return words

def create_lookup_tables(words):
    """
    Create lookup tables for vocabulary
    :param words: Input list of words
    :return: A tuple of dicts.  The first dict maps a vocab word to and integeter
             The second maps an integer back to to the vocab word
    """
    word_counts = Counter(words)
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    int_to_vocab = {ii: word for ii, word in enumerate(sorted_vocab, 1)}
    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}

    return vocab_to_int, int_to_vocab

def encode_ST_messages(messages, vocab_to_int):
    """
    Encode ST Sentiment Labels
    :param messages: list of list of strings. List of message tokens
    :param vocab_to_int: mapping of vocab to idx
    :return: list of ints. Lists of encoded messages
    """
    messages_encoded = []
    for message in messages:
        messages_encoded.append([vocab_to_int[word] for word in message.split()])

    return np.array(messages_encoded)

def encode_ST_labels(labels):
    """
    Encode ST Sentiment Labels
    :param labels: Input list of labels
    :return: numpy array.  The encoded labels
    """
    return np.array([1 if sentiment == 'bullish' else 0 for sentiment in labels])

def drop_empty_messages(messages, labels):
    """
    Drop messages that are left empty after preprocessing
    :param messages: list of encoded messages
    :return: tuple of arrays. First array is non-empty messages, second array is non-empty labels
    """
    non_zero_idx = [ii for ii, message in enumerate(messages) if len(message) != 0]
    messages_non_zero = np.array([messages[ii] for ii in non_zero_idx])
    labels_non_zero = np.array([labels[ii] for ii in non_zero_idx])
    return messages_non_zero, labels_non_zero

def zero_pad_messages(messages, seq_len):
    """
    Zero Pad input messages
    :param messages: Input list of encoded messages
    :param seq_ken: Input int, maximum sequence input length
    :return: numpy array.  The encoded labels
    """
    messages_padded = np.zeros((len(messages), seq_len), dtype=int)
    for i, row in enumerate(messages):
        messages_padded[i, -len(row):] = np.array(row)[:seq_len]

    return np.array(messages_padded)

def train_val_test_split(messages, labels, split_frac, random_seed=None):
    """
    Zero Pad input messages
    :param messages: Input list of encoded messages
    :param labels: Input list of encoded labels
    :param split_frac: Input float, training split percentage
    :return: tuple of arrays train_x, val_x, test_x, train_y, val_y, test_y
    """
    # make sure that number of messages and labels allign
    assert len(messages) == len(labels)
    # random shuffle data
    if random_seed:
        np.random.seed(random_seed)
    shuf_idx = np.random.permutation(len(messages))
    messages_shuf = np.array(messages)[shuf_idx] 
    labels_shuf = np.array(labels)[shuf_idx]

    #make splits
    split_idx = int(len(messages_shuf)*split_frac)
    train_x, val_x = messages_shuf[:split_idx], messages_shuf[split_idx:]
    train_y, val_y = labels_shuf[:split_idx], labels_shuf[split_idx:]

    test_idx = int(len(val_x)*0.5)
    val_x, test_x = val_x[:test_idx], val_x[test_idx:]
    val_y, test_y = val_y[:test_idx], val_y[test_idx:]

    return train_x, val_x, test_x, train_y, val_y, test_y
    
def get_batches(x, y, batch_size=100):
    """
    Batch Generator for Training
    :param x: Input array of x data
    :param y: Input array of y data
    :param batch_size: Input int, size of batch
    :return: generator that returns a tuple of our x batch and y batch
    """
    n_batches = len(x)//batch_size
    x, y = x[:n_batches*batch_size], y[:n_batches*batch_size]
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii+batch_size], y[ii:ii+batch_size]