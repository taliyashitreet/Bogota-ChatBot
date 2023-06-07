from sentence_transformers import SentenceTransformer
import torch
import tensorflow as tf
import numpy as np
import re
import sys
from keras.layers import TextVectorization
from tensorflow.python.keras import metrics

class_weights_tensor = np.load('class_weights_tensor.npy')

def weighted_binary_crossentropy(y_true, y_pred):
    return tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, class_weights_tensor)

class F1Score(metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision = metrics.Precision()
        self.recall = metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()

    def result(self):
        precision = self.precision.result()
        recall = self.recall.result()
        return 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the saved pytorch model
ner_model = torch.load('Model_3.pth', map_location=device)

# Load the saved tokenizer for NER
tokenizer = torch.load('Our_Token.pth', map_location=device)

# Load TF-IDF category model
category_model = tf.keras.models.load_model('TFIDF__Model.h5', custom_objects={'F1Score': F1Score, 'weighted_binary_crossentropy': weighted_binary_crossentropy})

# Load BERT Model
sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
# Load BERT category model
bert_category_model = tf.keras.models.load_model('BERT_20230606.h5', custom_objects={'F1Score': F1Score})

# Load the vocabulary from a file
with open('vocabulary.txt', 'r') as f:
    vocabulary = [line.strip() for line in f]

# Load the categories from a file
with open('categories.txt', 'r') as f:
    categories = [line.strip() for line in f]

# Create a new TextVectorization layer and adapt it using the saved vocabulary
text_vectorizer = TextVectorization(max_tokens=len(vocabulary), ngrams=2, output_mode="tf_idf")
text_vectorizer.adapt(vocabulary)
tag_values = ['WP', 'VBG', 'RRB', 'IN', 'JJ', 'PDT', 'NNPS', 'VBZ', 'RB', 'VBD', 'EX', 'JJS', 'LRB', 'FW', 'CC', '.',
              'JJR', 'NNP', 'VBN', 'CD', 'NNS', 'DT', 'VB', 'POS', 'WDT', 'MD', '$', 'RP', ',', 'PRP', 'VBP', 'NN', ':',
              'PRP$', 'RBS', 'UH', 'WRB', 'WP$', '``', 'RBR', ';', 'TO', 'PAD']

# For the unsupervised method

# Loading models from tfhub.dev
encoder = hub.KerasLayer("https://tfhub.dev/jeongukjae/smaller_LaBSE_15lang/1")
preprocessor = hub.KerasLayer("https://tfhub.dev/jeongukjae/smaller_LaBSE_15lang_preprocess/1")

# Constructing model to encode texts into high-dimensional vectors
sentences = tf.keras.layers.Input(shape=(), dtype=tf.string, name="sentences")
encoder_inputs = preprocessor(sentences)
sentence_representation = encoder(encoder_inputs)["pooled_output"]
normalized_sentence_representation = tf.nn.l2_normalize(sentence_representation, axis=-1)  # for cosine similarity
model = tf.keras.Model(sentences, normalized_sentence_representation)


index_category = {0:'Environment and climate resilience',1:'Mobility (transport)',2:'Local identity',3:'Future of work',4:'Land use'}


def predict_category_all_sentence(sentence: str):
    TRESHOLD = 0.25

    # Encoding the messages and the categories sentences.
    messages_sentences = tf.constant([sentence])
    categories_sentences = tf.constant(
        ["Environment and climate resilience", "Mobility (transport)", "Local identity", "Future of work", "Land use"])

    messages_embeds = model(messages_sentences)
    categories_embeds = model(categories_sentences)

    # Messages-categories similarity
    result = tf.tensordot(messages_embeds, categories_embeds, axes=[[1], [1]])

    category = ''
    counter = 0
    for value in result:  # result = [[3432 34234 234 324234 23]]
        for i, v in enumerate(value):  # for each number in the list
            if float(v) > TRESHOLD:  # needs to be change accorindg to the result from ChatGPT
                if counter > 0:
                    category += ', ' + index_category.get(i)
                else:
                    category += index_category.get(i)
                    counter += 1

    if category == '':
        return 'Other'
    return category


def predict_category_concatenated_nouns(nouns: list, test_sentence: str):
    TRESHOLD = 0.27

    conc_str = ' '.join(nouns)

    # Encoding the messages and the categories sentences.
    messages_sentences = tf.constant([conc_str])
    categories_sentences = tf.constant(
        ["Environment and climate resilience", "Mobility (transport)", "Local identity", "Future of work", "Land use"])

    messages_embeds = model(messages_sentences)
    categories_embeds = model(categories_sentences)

    # Messages-categories similarity
    result = tf.tensordot(messages_embeds, categories_embeds, axes=[[1], [1]])

    category = ''
    counter = 0
    for value in result:  # result = [[3432 34234 234 324234 23]]
        for i, v in enumerate(value):  # for each number in the list
            if float(v) > TRESHOLD:
                if counter > 0:
                    category += ', ' + index_category.get(i)
                else:
                    category += index_category.get(i)
                    counter += 1

    if category == '':
        return 'Other'
    return category


def predict_category_avg_category_nouns(nouns: list, test_sentence: str):
    TRESHOLD = 0.25

    sum_result_column = [0 for i in range(5)]

    for noun in nouns:

        # Encoding the messages and the categories sentences.
        messages_sentences = tf.constant([noun])
        categories_sentences = tf.constant(
            ["Environment and climate resilience", "Mobility (transport)", "Local identity", "Future of work",
             "Land use"])

        messages_embeds = model(messages_sentences)
        categories_embeds = model(categories_sentences)

        # Messages-categories similarity
        result = tf.tensordot(messages_embeds, categories_embeds, axes=[[1], [1]])

        for value in result:  # result = [[3432 34234 234 324234 23]]
            for i, v in enumerate(value):  # for each number in the list
                sum_result_column[i] += float(v)

    sum_result_column = [num / 5 for num in sum_result_column]  # calac avg

    category = ''
    counter = 0
    for i, value in enumerate(sum_result_column):
        if float(value) > TRESHOLD:
            if counter > 0:
                category += ', ' + index_category.get(i)
            else:
                category += index_category.get(i)
                counter += 1

    if category == '':
        return 'Other'
    return category


def test_model(test_sentence):
    tokenized_sentence = tokenizer.encode(test_sentence)
    input_ids = torch.tensor([tokenized_sentence]).to(device)
    with torch.no_grad():
        output = ner_model(input_ids)
    label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)
    tokens = tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])
    new_tokens, new_labels = [], []
    for token, label_idx in zip(tokens, label_indices[0]):
        if token.startswith("##"):
            new_tokens[-1] = new_tokens[-1] + token[2:]
        else:
            new_labels.append(tag_values[label_idx])
            new_tokens.append(token)
    ans = ""
    for token, label in zip(new_tokens, new_labels):
        ans += "{}\t{}".format(label, token)
        ans += "\n"
    nouns = re.findall(r'NN\w*\s+(\w+)', ans)
    return nouns


def predict_TFIDF(sentence):
    tokenized_sentence = text_vectorizer([sentence])
    category_vector = category_model.predict(tokenized_sentence)

    # Apply the logic from custom_round to category_vector
    binary_category_vector = np.zeros_like(category_vector)
    for i, row in enumerate(category_vector):
        if np.all(row < 0.5):  # this threshold should be set to the same value used in custom_round
            binary_category_vector[i, np.argmax(row)] = 1
        else:
            binary_category_vector[i] = np.round(row)

    # Find the indices of the categories that were predicted
    category_indices = np.where(binary_category_vector[0] == 1)[0]
    category_names = [categories[i] for i in category_indices]

    return category_names

def invert_multi_hot(encoded_labels):
    """Reverse a single multi-hot encoded label to a list of category names."""
    hot_indices = np.argmax(encoded_labels, axis=-1)
    return [categories[idx] for idx in hot_indices]

def predict_category_bert(sentence):
    sentence_embedding = sbert_model.encode([sentence])
    category_vector = bert_category_model.predict(sentence_embedding)

    # Apply the logic to retrieve category predictions
    binary_category_vector = np.zeros_like(category_vector)
    for i, row in enumerate(category_vector):
        if np.all(row < 0.5):  # this threshold should be set to the same value used in custom_round
            binary_category_vector[i, np.argmax(row)] = 1
        else:
            binary_category_vector[i] = np.round(row)

    # Convert the binary predictions to their actual names
    predicted_labels = [invert_multi_hot(pred) for pred in binary_category_vector]
    return predicted_labels[0]



def process_file(input_file_path):
    with open(input_file_path, 'r') as file:
        test_sentence = file.read()
    nouns = test_model(test_sentence)
    
    categories_TFIDF = predict_TFIDF(test_sentence)
    categories_BERT = predict_category_bert(test_sentence)

    # unsupervied
    categories_sent = predict_category_all_sentence(test_sentence)
    categories_conc_nouns = predict_category_concatenated_nouns(nouns, test_sentence)
    categories_nouns_avg = predict_category_avg_category_nouns(nouns, test_sentence)

    print("The categories of this sentence according to BERT: " + ', '.join(categories_BERT))
    with open(input_file_path, 'a') as file:
        file.write("\nThe sentence's nouns are: " + ', '.join(nouns) + "\n")
        file.write("The categories of this sentence according to TF-IDF: " + ', '.join(categories_TFIDF) + "\n")
        file.write("The categories of this sentence according to BERT: " + ', '.join(categories_BERT) + "\n")
        file.write("The categories of this sentence according to unsuper-sentence: " + categories_sent + "\n")
        file.write("The categories of this sentence according to unsuper-conc-nouns: " + categories_conc_nouns + "\n")
        file.write("The categories of this sentence according to unsuper-nouns-avg: " + categories_nouns_avg + "\n")

input_file_path = sys.argv[1]
process_file(input_file_path)
