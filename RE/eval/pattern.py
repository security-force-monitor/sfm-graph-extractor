import os, pickle, sys
sys.path.insert(1, '../4. dep+np')
from path import *
from eval import *

pred_path = "../4. dep+np/out"
true_path = "ann/truth"

if __name__ == "__main__":
    pred_filenames = os.listdir(pred_path)
    true_filenames = os.listdir(true_path)

    pred_doc_ids = []
    for fn in pred_filenames:
        if fn[-3:] == "ann":
            with open(os.path.join(pred_path, fn), 'r') as pfile:
                # pred_docs[fn] = pfile.read()
                pred_doc_ids.append(fn[:-4])

    path_pattern = {}
    path_pattern_reverse = {}
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

            for per_id in persons_nodes:
                for ot_id in others_nodes:
                    for rl in true_relations:
                        if rl.arg1.id == per_id and rl.arg2.id == ot_id:
                            # print('-------------------', rl)
                            path = get_ne_path(parse_tree, persons_nodes[per_id], others_nodes[ot_id])
                            path_types = get_type_path(parse_tree, path)

                            if rl.arg2.type in path_pattern:
                                if path_types in path_pattern[rl.arg2.type]:
                                    path_pattern[rl.arg2.type][path_types] += 1
                                else:
                                    path_pattern[rl.arg2.type][path_types] = 1
                            else:
                                path_pattern[rl.arg2.type] = {path_types : 1}

                            if path_types in path_pattern_reverse:
                                if rl.arg2.type in path_pattern_reverse[path_types]:
                                    path_pattern_reverse[path_types][rl.arg2.type] += 1
                                else:
                                    path_pattern_reverse[path_types][rl.arg2.type] = 1
                            else:
                                path_pattern_reverse[path_types] = {rl.arg2.type : 1}

            sentence_start += len(all_lines[line_count]) + 1
            line_count += 1

    # pp.pprint(path_pattern)
    # pp.pprint(path_pattern_reverse)

    pp_filename = "path_dict.p"
    pp_filename_reverse = "path_dict_reverse.p"
    pickle.dump(path_pattern, open(pp_filename, "wb"))
    pickle.dump(path_pattern_reverse, open(pp_filename_reverse, "wb"))


    path_pattern = pickle.load(open(pp_filename, "rb"))
    path_pattern_reverse = pickle.load(open(pp_filename_reverse, "rb"))

    threshold = 3

    new_path_pattern = {}
    for type in path_pattern:
        new_path_pattern[type] = {}
        for pattern in path_pattern[type]:
            if path_pattern[type][pattern] >= threshold:
                new_path_pattern[type][pattern] = path_pattern[type][pattern]

    pp.pprint(new_path_pattern)

    new_path_pattern_reverse = {}
    for pattern in path_pattern_reverse:
        cur_dict = {}
        for type in path_pattern_reverse[pattern]:
            if path_pattern_reverse[pattern][type] >= threshold:
                cur_dict[type] = path_pattern_reverse[pattern][type]
        if cur_dict:
            new_path_pattern_reverse[pattern] = cur_dict

    pp.pprint(new_path_pattern_reverse)
