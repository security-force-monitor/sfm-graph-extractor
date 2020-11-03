import os, sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras import layers
from train_re import PatTypeOH

from config import *
from data import *
from ne_def import *
import parse

def get_person_nodes(parse_tree, all_persons, sentence_span):
    person_nodes = {} # ne id --> node ids
    for idx, per in enumerate(all_persons):
        if sentence_span[0] <= per.span[0] and per.span[1] < sentence_span[1]:
            found_nodes = []
            for node in parse_tree:
                if node[1] in per.name.split():
                    found_nodes.append(node[0])
            if len(found_nodes) > 0:
                person_nodes[per.id] = found_nodes
    return person_nodes

def is_diff_stn(stn_split_punc, txt):
    for punc in stn_split_punc:
        if punc in txt:
            return True
    return False


if __name__ == "__main__":

    package_path, package_init = os.path.split(__file__)
    all_patterns = pickle.load(open(os.path.join(package_path, patterns_filename), "rb"))
    new_model = keras.models.load_model(os.path.join(package_path, model_path), custom_objects={'PatTypeOH': PatTypeOH})

    if len(sys.argv) < 2:
        print("Type in the path to the input directory (use absolute path).")
        exit()

    input_dir = sys.argv[1]

    pred_filenames = os.listdir(input_dir)
    pred_doc_ids = []
    for fn in pred_filenames:
        if fn[-3:] == "ann":
            with open(os.path.join(input_dir, fn), 'r') as pfile:
                pred_doc_ids.append(fn[:-4])

    dataset = []
    for doc_id in pred_doc_ids:
        print(doc_id)

        buffer_path = os.path.join(input_dir, doc_id)
        txt_path = os.path.join(input_dir, doc_id + ".txt")
        ann_path = os.path.join(input_dir, doc_id + ".ann.nn")

        all_persons, all_others, _ = get_ne_rl(ann_path)
        all_lines = get_lines(txt_path)
        with open(txt_path) as input_file:
            whole_doc = input_file.read()

        line_count = 0
        rel_count = 0
        sentence_start = 0
        while os.path.isfile( os.path.join(buffer_path, str(line_count) + '.txt.conllu.pred') ):
            if line_count >= len(all_lines):
                break
            sentence_start, sentence_end = correct_position(whole_doc, sentence_start, all_lines[line_count])

            parse_tree = parse.get_parse_tree(buffer_path, line_count)

            others_nodes = get_ne2node_mapping(parse_tree, all_others, sentence_start, 1)
            persons_nodes = get_ne2node_mapping(parse_tree, all_persons, sentence_start, 1)

            relations = []
            for ot_id in others_nodes:
                input = []
                per_ne_list = []

                left_person = None
                right_person = None

                for per_ne in all_persons:

                    ne = get_name_entity(ot_id, all_others)
                    if per_ne.span[0] >= sentence_start and per_ne.span[1] <= sentence_end:
                        if per_ne.span[1] <= ne.span[0]:
                            if left_person is None or per_ne.span[1] > left_person.span[1]:
                                left_person = per_ne
                        if per_ne.span[0] >= ne.span[1]:
                            if right_person is None or per_ne.span[0] < right_person.span[0]:
                                right_person = per_ne

                    if sentence_start <= per_ne.span[0] and per_ne.span[1] <= sentence_end:
                        path = get_ne_path(parse_tree, persons_nodes[per_ne.id], others_nodes[ot_id])
                        path_types = get_type_path(parse_tree, path)
                        input.append(path_types)
                        per_ne_list.append(per_ne)

                pt = {  "x": remove_low_freq_patterns(input, all_patterns),
                        "type": get_name_entity(ot_id, all_others).type }

                onehot_len_set, type_1hot = get_1hot_vecs(pt, all_patterns)
                X_pt = np.concatenate((np.reshape(onehot_len_set, (-1,)), np.array(type_1hot)), axis = 0)
                X_pt = np.expand_dims(X_pt, axis = 0)
                pred_output = new_model.predict(X_pt)
                per_ne_idx = np.argmax(pred_output)

                # print(per_ne_idx)
                best_per = None
                if per_ne_idx == 0 and left_person is not None:
                    best_per = left_person
                if per_ne_idx == 1 and right_person is not None:
                    best_per = right_person
                if per_ne_idx == 2:
                    for per_idx, per in enumerate(per_ne_list):
                        per_sentence_span = [per.span[0] - sentence_start, per.span[1] - sentence_start]
                        cur_dist = parse.get_entity_distance(parse_tree, ne_sentence_span, per_sentence_span)
                        if (left_person is not None and left_person.id != per.id) and (right_person is not None and right_person.id != per.id):
                            if cur_dist is not None and cur_dist < min_dist:
                                min_dist = cur_dist
                                best_per = per

                if best_per is not None:
                    ne = get_name_entity(ot_id, all_others)
                    if ne.type == "TOR":
                        rel_count += 1
                        relations.append(Relation(rel_count, best_per, ne, HAS_TOR))
                    elif ne.type == "RNK":
                        rel_count += 1
                        relations.append(Relation(rel_count, best_per, ne, HAS_RANK))
                    elif ne.type == "ORG":
                        rel_count += 1
                        relations.append(Relation(rel_count, best_per, ne, IS_POSTED))

                    # print(relations[-1])

            ann_file = open(ann_path, 'a')
            for rl in relations:
                ann_file.write(rl.get_ann_str() + '\n')
            ann_file.close()

            sentence_start += len(all_lines[line_count]) + 1
            line_count += 1
