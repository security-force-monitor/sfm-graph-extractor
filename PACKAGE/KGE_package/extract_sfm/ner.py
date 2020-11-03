import sys
import pred
from eval import *
from ne_def import *

ENDING_PUNCTUATION = ",."
SHIFT_RANGE = 3


def find_entity_within(query_en, en_list):
    for en in en_list:

        if (en.type == "RNK" or en.type == "TOR") and \
            en.name.lower() != query_en.name.lower() and \
            en.name.lower() in query_en.name.lower():
            en_start = query_en.name.lower().find(en.name.lower())
            en_end = en_start + len(en.name)
            if en_start == 0 or en_end == len(query_en.name):
                return en_start, en_end, en.type
            else:
                return None
    return None


if __name__ == '__main__':

    if len(sys.argv) < 2:
        print("Type in the name of the input data file")
        exit()

    doc_file = sys.argv[1]
    with open(doc_file) as test_file:
        whole_doc = test_file.read()
    with open(doc_file) as test_file:
        test_sentences = test_file.readlines()
    doc_sentences = [sentence.strip() for sentence in test_sentences if sentence.strip() != '' and not sentence.strip().isspace()]
    sentence_pred_tags = pred.build_pred_dict(doc_sentences)

    # Build a list of NameEntity instances for recognized entities.
    name_entity_count = 1
    cur_sentence_start = 0
    doc_entities = []
    for sentence in doc_sentences:
        cur_sentence_start, cur_sentence_end = correct_position(whole_doc, cur_sentence_start, sentence)

        pred_entities = []
        for name_position in sentence_pred_tags[sentence].keys():
            pred_name = sentence[name_position[0]: name_position[1]]
            pred_tag = sentence_pred_tags[sentence][name_position]
            doc_name_position = [cur_sentence_start + name_position[0], \
                                cur_sentence_start + name_position[1]]
            # Correct name entity span
            # Remove punctuations
            if pred_name[-1] in ENDING_PUNCTUATION:
                pred_name = pred_name[:-1]
                doc_name_position[1] -= 1
            # Correct name entity shift due to various spacing between lines
            if whole_doc[doc_name_position[0]: doc_name_position[1]] != pred_name:
                doc_name_start, doc_name_end = correct_position(whole_doc, doc_name_position[0], pred_name)
                doc_name_position = [doc_name_start, doc_name_end]

            pred_entities.append(NameEntity(name_entity_count, pred_name, pred_tag, doc_name_position))
            name_entity_count += 1
        doc_entities += pred_entities

        cur_sentence_start += len(sentence) + 2

    # Write .ann file
    ann_path = sys.argv[1][:-4] + '.ann'
    ann_file = open(ann_path,'w')
    ann_file.write("")
    for en in doc_entities:
        ann_file.write(en.get_ann_str() + '\n')
    ann_file.close()
