import tensorflow as tf
import torch
import numpy as np
import re

from keras.layers import TextVectorization
from tensorflow.python.keras.saving.save import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras import metrics
from tensorflow.keras import layers

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

# Load the saved tokenizer
tokenizer = torch.load('Our_Token.pth', map_location=device)


category_model = tf.keras.models.load_model('model_text_clf_tokenizer80.h5', custom_objects={'F1Score': F1Score})



# Load the vocabulary from a file
with open('vocabulary.txt', 'r') as f:
    vocabulary = [line.strip() for line in f]


# Load the categories from a file
with open('categories.txt', 'r') as f:
    categories = [line.strip() for line in f]

# Create a new TextVectorization layer and adapt it using the saved vocabulary
text_vectorizer = TextVectorization(max_tokens=len(vocabulary), ngrams=2, output_mode="tf_idf")
text_vectorizer.adapt(vocabulary)
tag_values = ['WP', 'VBG', 'RRB', 'IN', 'JJ', 'PDT', 'NNPS', 'VBZ', 'RB', 'VBD', 'EX', 'JJS', 'LRB', 'FW', 'CC', '.', 'JJR', 'NNP', 'VBN', 'CD', 'NNS', 'DT', 'VB', 'POS', 'WDT', 'MD', '$', 'RP', ',', 'PRP', 'VBP', 'NN', ':', 'PRP$', 'RBS', 'UH', 'WRB', 'WP$', '``', 'RBR', ';', 'TO', 'PAD']

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

def predict_category(sentence):
    tokenized_sentence = text_vectorizer([sentence])
    category_vector = category_model.predict(tokenized_sentence)
    category_indices = np.where(category_vector[0] > 0.5)[0]
    category_names = [categories[i] for i in category_indices]
    return category_names

def process_file(input_file_path):
    with open(input_file_path, 'r') as file:
        test_sentence = file.read()
    nouns = test_model(test_sentence)
    categories = predict_category(test_sentence)
    with open(input_file_path, 'a') as file:
        file.write("\nThe sentence's nouns are: " + ', '.join(nouns) + "\n")
        file.write("The categories of this sentence according to TF-IDF: " + ', '.join(categories) + "\n")

if __name__ == "__main__":
    import sys
    input_file_path = sys.argv[1]
    process_file(input_file_path)
