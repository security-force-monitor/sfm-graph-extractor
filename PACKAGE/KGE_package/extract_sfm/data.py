import os, pickle, sys
import numpy as np
from path import *
import parse
from ne_def import *
from config import *

np.set_printoptions(threshold=sys.maxsize)

import pprint
pp = pprint.PrettyPrinter(indent=2)

def get_ne_of_type(ne_list, type):
    ne_of_type = []
    for other in ne_list:
        if other.type == type:
            ne_of_type.append(other)
    return ne_of_type

def get_1hot_vecs(pt, all_patterns):
    onehot_len_set = []
    for pattern in pt["x"]:
        one_hot = [0,] * len(all_patterns)
        idx = all_patterns.index(pattern)
        one_hot[idx] = 1
        # one_hot_vecs.append(one_hot)
        # lengths.append(len(pattern))
        onehot_len = one_hot + [len(pattern)]
        onehot_len_set.append(onehot_len)

    # pt["one_hots"] = one_hot_vecs
    # pt["lengths"] = lengths
    if len(onehot_len_set) < PER_NUM:
        onehot_len_set += [get_pattern_padding(all_patterns)] * (PER_NUM - len(onehot_len_set))
    elif len(onehot_len_set) > PER_NUM:
        onehot_len_set = onehot_len_set[:PER_NUM]

    type_1hot = np.zeros(len(ALL_NE_TYPES))
    type_1hot[ALL_NE_TYPES.index(pt['type'])] = 1
    type_1hot = type_1hot.tolist()

    return onehot_len_set, type_1hot

def get_pattern_padding(all_patterns):
    return [0,] * (len(all_patterns) + 1)

def remove_low_freq_patterns(pat_list, all_patterns):
    freq_patterns = []
    for pattern in pat_list:
        if pattern in all_patterns:
            freq_patterns.append(pattern)
        else:
            freq_patterns.append(UNKNOWN)
    return freq_patterns


if __name__ == "__main__":
    pred_filenames = os.listdir(pred_path)
    true_filenames = os.listdir(true_path)

    pred_doc_ids = []
    for fn in pred_filenames:
        if fn[-3:] == "ann":
            with open(os.path.join(pred_path, fn), 'r') as pfile:
                # pred_docs[fn] = pfile.read()
                pred_doc_ids.append(fn[:-4])


    dataset = []
    for doc_id in pred_doc_ids:
        buffer_path = os.path.join(pred_path, doc_id)
        txt_path = os.path.join(pred_path, doc_id + ".txt")
        true_ann_path = os.path.join(true_path, doc_id + ".ann")

        with open(txt_path) as input_file:
            whole_doc = input_file.read()

        all_lines = get_lines(txt_path)
        true_persons, true_others, true_relations = get_ne_rl(true_ann_path)

        line_count = 0
        rel_count = 0
        sentence_start = 0
        while os.path.isfile( os.path.join(buffer_path, str(line_count) + '.txt.conllu.pred') ):
            if line_count >= len(all_lines):
                break
            sentence_start, sentence_end = correct_position(whole_doc, sentence_start, all_lines[line_count])

            parse_tree = parse.get_parse_tree(buffer_path, line_count)

            others_nodes = get_ne2node_mapping(parse_tree, true_others, sentence_start, 1)
            persons_nodes = get_ne2node_mapping(parse_tree, true_persons, sentence_start, 1)

            for ot_id in others_nodes:
                input = []
                target_cat = 2

                left_person = None
                right_person = None
                for per_ne in get_ne_of_type(true_persons , "PER"):
                    ne = get_name_entity(ot_id, true_others)
                    if per_ne.span[0] >= sentence_start and per_ne.span[1] <= sentence_end:
                        if per_ne.span[1] <= ne.span[0]:
                            if left_person is None or per_ne.span[1] > left_person.span[1]:
                                left_person = per_ne
                        if per_ne.span[0] >= ne.span[1]:
                            if right_person is None or per_ne.span[0] < right_person.span[0]:
                                right_person = per_ne

                    for rl in true_relations:
                        if rl.arg2.id == ot_id:
                            if left_person is not None and rl.arg1.id == left_person.id:
                                # path = get_ne_path(parse_tree, persons_nodes[left_person.id], others_nodes[ot_id])
                                # type_path = get_type_path(parse_tree, path)
                                # input.append(type_path)
                                target_cat = 0
                                # type = rl.arg2.type

                            if right_person is not None and rl.arg1.id == right_person.id:
                                # path = get_ne_path(parse_tree, persons_nodes[right_person.id], others_nodes[ot_id])
                                # type_path = get_type_path(parse_tree, path)
                                # input.append(type_path)
                                target_cat = 1
                                # type = rl.arg2.type

                    if sentence_start <= per_ne.span[0] and per_ne.span[1] <= sentence_end:
                        per_id = per_ne.id
                        for rl in true_relations:
                            if rl.arg1.id == per_id and rl.arg2.id == ot_id:
                                path = get_ne_path(parse_tree, persons_nodes[per_id], others_nodes[ot_id])
                                path_types = get_type_path(parse_tree, path)
                                input.append(path_types)
                                # target.append(1)
                                type = rl.arg2.type
                            elif rl.arg2.id == ot_id:
                                path = get_ne_path(parse_tree, persons_nodes[per_id], others_nodes[ot_id])
                                path_types = get_type_path(parse_tree, path)
                                input.append(path_types)
                                # target.append(0)
                                type = rl.arg2.type

                target = [0, 0, 0]
                target[target_cat] = 1

                pt = {  "x": input,
                        "y": target,
                        "type": type }
                dataset.append(pt)

            sentence_start += len(all_lines[line_count]) + 1
            line_count += 1

    # pp_filename = "data.p"
    # pickle.dump(dataset, open(pp_filename, "wb"))
    #
    # pp_filename = "data.p"
    # dataset = pickle.load(open(pp_filename, "rb"))

    pattern_freqs = {}
    for pt in dataset:
        # print(pt)
        for pattern in pt["x"]:
            if pattern in pattern_freqs:
                pattern_freqs[pattern] += 1
            else:
                pattern_freqs[pattern] = 1

    percentile_threshold = np.percentile(list(pattern_freqs.values()), PERCENTILE)
    all_patterns = [UNKNOWN,]
    for idx, pt in enumerate(dataset):
        for pattern in pt["x"]:
            if pattern in pattern_freqs \
                and pattern_freqs[pattern] > percentile_threshold \
                and pattern not in all_patterns:
                    all_patterns.append(pattern)


    print("One hot vector of input has length: ", len(all_patterns))

    for idx, pt in enumerate(dataset):
        dataset[idx]['x'] = remove_low_freq_patterns(pt['x'], all_patterns)
        onehot_len_set, type_1hot = get_1hot_vecs(pt, all_patterns)
        dataset[idx]["1hot_len"] = onehot_len_set
        dataset[idx]["type_oh"] = type_1hot

    X_data = []
    X_type = []
    Y_data = []
    # max_y_len = 0
    for pt in dataset:
        X_data.append(pt["1hot_len"])
        X_type.append(pt["type_oh"])
        y_pt = pt["y"]
        # if len(y_pt) > max_y_len:
        #     max_y_len = len(y_pt)
        if len(y_pt) < PER_NUM:
            y_pt += [0,] * (PER_NUM - len(y_pt))
        elif len(y_pt) > PER_NUM:
            y_pt = y_pt[:PER_NUM]
        Y_data.append(y_pt)

    X_data = np.array(X_data)
    X_data = np.reshape(X_data, (X_data.shape[0], -1))
    X_data = np.concatenate((X_data, np.array(X_type)), axis = 1)
    Y_data = np.array(Y_data)

    print("X shape: ", X_data.shape)
    print("Y shape: ", Y_data.shape)

    pickle.dump([X_data, Y_data], open(dataset_filename, "wb"))
    pickle.dump(all_patterns, open(patterns_filename, "wb"))
