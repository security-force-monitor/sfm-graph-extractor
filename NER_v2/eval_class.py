import pickle
import string
from difflib import SequenceMatcher
import process
import os
import pprint
pp = pprint.PrettyPrinter(indent=2)
import spacy
nlp = spacy.load("en_core_web_sm")
from eval import *
from ne_def import label_mapping, inv_label_mapping

pickle_path = './SFM_STARTER'

if __name__ == '__main__':
    with open(os.path.join(pickle_path, "dataset_sentences.pickle"),"rb") as pickle_file:
        dataset_sentences = pickle.load(pickle_file)
    with open(os.path.join(pickle_path, "dataset_labels.pickle"),"rb") as pickle_file:
        dataset_labels = pickle.load(pickle_file)
    with open("sentence_pred_tags.pickle","rb") as pickle_in:
        sentence_pred_tags = pickle.load(pickle_in)
    with open(os.path.join(pickle_path, 'test.words.txt'), 'r') as test_text:
        test_lines = test_text.readlines()

    true_positive_count = {"PER": 0, "RNK": 0, "ORG": 0, "TOR": 0, "LOC": 0}
    false_positive_count = {"PER": 0, "RNK": 0, "ORG": 0, "TOR": 0, "LOC": 0}
    false_negative_count = {"PER": 0, "RNK": 0, "ORG": 0, "TOR": 0, "LOC": 0}
    false_entities = {"PER": [[], []], "RNK": [[], []], "ORG": [[], []], "TOR": [[], []], "LOC": [[], []]} # [list of fp, list of fn]
    # similar_true_positive_count = {"PER": 0, "RAN": 0, "ORG": 0, "TIT": 0, "ROL": 0, "LOC": 0}
    # similar_false_positive_count = {"PER": 0, "RAN": 0, "ORG": 0, "TIT": 0, "ROL": 0, "LOC": 0}
    # similar_false_negative_count = {"PER": 0, "RAN": 0, "ORG": 0, "TIT": 0, "ROL": 0, "LOC": 0}
    for line in test_lines:
        sentence = line.strip()
        id, s_position = dataset_sentences[sentence]
        print('=================================================')
        print("From file with id: ", id)
        print("Sentence: ")
        pp.pprint(sentence)

        print('\n-------------- Ground truth --------------')
        ground_truth_names = []
        ground_truth_tags = []
        for label in dataset_labels[id][s_position]:
            gt_name = process.get_name(sentence, label)
            ground_truth_names.append(tokenizer(gt_name))
            ground_truth_tags.append(label_mapping[label[1]])
            print(ground_truth_names[-1], "||||", label[1], "||||", label[2])
        # print(ground_truth_names)

        print('\n-------------- Predicted labels --------------')
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

        # print('\n--------------False entities--------------')
        print('\n>>> False-positives:')
        for idx, pred_name in enumerate(pred_names):
            if pred_name in ground_truth_names:
                true_tag = ground_truth_tags[ground_truth_names.index(pred_name)]
                if true_tag == pred_tags[idx]:
                    true_positive_count[pred_tags[idx]] += 1
                    continue

            false_positive_count[pred_tags[idx]] += 1
            print("\t", pred_name, "||||", inv_label_mapping[pred_tags[idx]])
            false_entities[pred_tags[idx]][0].append((pred_name, pred_tags[idx]))

        print('>>> False-negatives:')
        for true_name in ground_truth_names:
            true_tag = ground_truth_tags[ground_truth_names.index(true_name)]
            if true_name not in pred_names:
                false_negative_count[true_tag] += 1
                print("\t", true_name, "||||", inv_label_mapping[true_tag])
                false_entities[true_tag][1].append((true_name, true_tag))
            else:
                pred_tag = pred_tags[pred_names.index(true_name)]
                if true_tag != pred_tag:
                    false_negative_count[true_tag] += 1
                    print("\t", true_name, "||||", inv_label_mapping[true_tag])
                    false_entities[true_tag][1].append((true_name, true_tag))


        # for pred_name in pred_names:
        #     has_similar = False
        #     for true_name in ground_truth_names:
        #         if similar(true_name, pred_name) > 0.5:
        #             has_similar = True
        #             break
        #     if has_similar:
        #         similar_true_positive_count[pred_tags[-1]] += 1
        #     else:
        #         similar_false_positive_count[pred_tags[-1]] += 1
        #
        # for true_name in ground_truth_names:
        #     has_similar = False
        #     for pred_name in pred_names:
        #         if similar(true_name, pred_name) > 0.7:
        #             has_similar = True
        #             break
        #     if not has_similar:
        #         similar_false_negative_count[pred_tags[-1]] += 1

        print('\n\n')

    # pp.pprint(true_positive_count)
    # pp.pprint(false_positive_count)
    # pp.pprint(false_negative_count)
    # pp.pprint(similar_true_positive_count)
    # pp.pprint(similar_false_positive_count)
    # pp.pprint(similar_false_negative_count)

    print("+--------------------------------------------------+")
    print("|               Precision and Recall               |")
    print("+--------------------------------------------------+")

    for tag in inv_label_mapping.keys():
        if tag == "LOC" or tag == "MISC":
            continue
        print("\n=============== Class: ", inv_label_mapping[tag], "===============")
        print("\ttp, fp, fn counts: ", [true_positive_count[tag], false_positive_count[tag], false_negative_count[tag]])
        precision = true_positive_count[tag] / (false_positive_count[tag] + true_positive_count[tag])
        recall = true_positive_count[tag] / (false_negative_count[tag] + true_positive_count[tag])
        f1_score = 2. / (1. / precision + 1. / recall)
        print('\tPrecision: ', precision)
        print('\tRecall: ', recall)
        print('\tF1 score: ', f1_score)
        # print('\n\t>>> False positive:')
        # pp.pprint(false_entities[tag][0])
        # print('\n\t>>> False negative:')
        # pp.pprint(false_entities[tag][1])
        # similar_precision = similar_true_positive_count[tag] / (similar_true_positive_count[tag] + similar_false_positive_count[tag])
        # similar_recall = similar_true_positive_count[tag] / (similar_true_positive_count[tag] + similar_false_negative_count[tag])
        # similar_f1_score = 2. / (1. / similar_precision + 1. / similar_recall)
        # print('Similar Precision: ', similar_precision)
        # print('Similar Recall: ', similar_recall)
        # print('Similar F1 score: ', similar_f1_score)
    print("\n=============== All classes ===============")
    precision = sum(true_positive_count.values()) / (sum(false_positive_count.values()) + sum(true_positive_count.values()))
    recall = sum(true_positive_count.values()) / (sum(false_negative_count.values()) + sum(true_positive_count.values()))
    f1_score = 2. / (1. / precision + 1. / recall)
    print('\tPrecision: ', precision)
    print('\tRecall: ', recall)
    print('\tF1 score: ', f1_score)
