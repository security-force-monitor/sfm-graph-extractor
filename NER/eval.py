import pickle
import string
from difflib import SequenceMatcher
import process
import os
import pprint
pp = pprint.PrettyPrinter(indent=2)
import spacy
nlp = spacy.load("en_core_web_sm")
from ne_def import inv_label_mapping

pickle_path = './SFM_STARTER'

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def tokenizer(name):
    doc = nlp(name)
    tokens = []
    for token in doc:
        if token.text not in string.punctuation:
            tokens.append(token.text)
    return ' '.join(tokens)

if __name__ == '__main__':
    with open(os.path.join(pickle_path, "dataset_sentences.pickle"),"rb") as pickle_file:
        dataset_sentences = pickle.load(pickle_file)
    with open(os.path.join(pickle_path, "dataset_labels.pickle"),"rb") as pickle_file:
        dataset_labels = pickle.load(pickle_file)
    with open("sentence_pred_tags.pickle","rb") as pickle_in:
        sentence_pred_tags = pickle.load(pickle_in)
    with open(os.path.join(pickle_path, 'test.words.txt'), 'r') as test_text:
        test_lines = test_text.readlines()

    true_positive_count = 0
    false_positive_count = 0
    false_negative_count = 0
    similar_true_positive_count = 0
    similar_false_positive_count = 0
    similar_false_negative_count = 0
    for line in test_lines:
        sentence = line.strip()
        id, s_position = dataset_sentences[sentence]
        print('=================================================')
        print("From file with id: ", id)
        print("Sentence: ")
        pp.pprint(sentence)

        print('\n--------------Ground truth--------------')
        ground_truth_names = []
        for label in dataset_labels[id][s_position]:
            gt_name = process.get_name(sentence, label)
            ground_truth_names.append(tokenizer(gt_name))
            print(ground_truth_names[-1], "||||", label[1], "||||", label[2])
        # print(ground_truth_names)

        print('\n--------------Predicted labels--------------')
        pred_names = []
        pred_tags = []
        for name_position in sentence_pred_tags[sentence].keys():
            pred_name = sentence[name_position[0]: name_position[1]]
            pred_names.append(tokenizer(pred_name))
            # pred_names.append(pred_name)
            pred_tags.append(sentence_pred_tags[sentence][name_position])
            print(pred_names[-1], "||||", inv_label_mapping[pred_tags[-1]], "||||", name_position)
            # exclude = set(string.punctuation)
            # pred_name_stripped = ''.join(ch for ch in pred_name if ch not in exclude)
            # pred_names.append(pred_name_stripped)

        for pred_name in pred_names:
            if pred_name in ground_truth_names:
                true_positive_count += 1
            else:
                false_positive_count += 1

        for true_name in ground_truth_names:
            if true_name not in pred_names:
                false_negative_count += 1


        for pred_name in pred_names:
            has_similar = False
            for true_name in ground_truth_names:
                if similar(true_name, pred_name) > 0.5:
                    has_similar = True
                    # print('-----------------')
                    # print(sentence)
                    # print('Ground Truth: ', true_name)
                    # print('Predicted Label: ', pred_name)
                    break
            if has_similar:
                similar_true_positive_count += 1
            else:
                # print('-----------------')
                # print(sentence)
                # print('Predicted Label: ', pred_name)
                # print('True Labels: ', end = '')
                # for true_name in ground_truth_names:
                #     print(true_name, end = ', ')
                # print('')
                similar_false_positive_count += 1

        for true_name in ground_truth_names:
            has_similar = False
            for pred_name in pred_names:
                if similar(true_name, pred_name) > 0.7:
                    has_similar = True
                    break
            if not has_similar:
                # print('-----------------')
                # print(sentence)
                # print('Ground Truth: ', true_name)
                # print('Predicted Labels: ', end = '')
                # for pname in pred_names:
                #     print(pname, end = ', ')
                # print('')
                similar_false_negative_count += 1

        print('\n\n')

    print('\n>>>')
    precision = true_positive_count / (false_positive_count + true_positive_count)
    recall = true_positive_count / (false_negative_count + true_positive_count)
    f1_score = 2. / (1. / precision + 1. / recall)
    print('Precision: ', precision)
    print('Recall: ', recall)
    print('F1 score: ', f1_score)

    similar_precision = similar_true_positive_count / (similar_true_positive_count + similar_false_positive_count)
    similar_recall = similar_true_positive_count / (similar_true_positive_count + similar_false_negative_count)
    similar_f1_score = 2. / (1. / similar_precision + 1. / similar_recall)
    print('Similar Precision: ', similar_precision)
    print('Similar Recall: ', similar_recall)
    print('Similar F1 score: ', similar_f1_score)
