import os
import sys
sys.path.insert(1, '../../NER_v2')
from ne_def import *
sys.path.insert(1, '../utils')
from path import *

pred_path = "../3. nn/out"
# pred_path = "dep"
true_path = "ann/truth"


def process_relation_str(line):
    line_split = line.split()
    rl_id = int(line_split[0][1:])
    type = line_split[1]
    arg1_id = int(line_split[2].split(':')[1][1:])
    arg2_id = int(line_split[3].split(':')[1][1:])
    return rl_id, type, arg1_id, arg2_id

def is_similar(name1, name2):
    return name1 in name2 or name2 in name1

def print_prl_trl(prl, trl):
    print("-----------------------------------")
    print("--- Pred:", prl, "||||", prl.arg1.span, "-->", prl.arg2.span)
    print("--- True:", trl, "||||", trl.arg1.span, "-->", trl.arg2.span)
    # print(prl.arg1.span, "||||", trl.arg1.span)
    # print(prl.arg2.span, "||||", trl.arg2.span)

def overlap_len(span_1, span_2):
    left_max = max(span_1[0], span_2[0])
    right_min = min(span_1[1], span_2[1])
    return right_min - left_max

def is_overlap(span_1, span_2):
    return overlap_len(span_1, span_2) >= 1

if __name__ == "__main__":
    pred_filenames = os.listdir(pred_path)
    true_filenames = os.listdir(true_path)

    pred_docs = {}
    for fn in pred_filenames:
        if fn[-3:] == "ann":
            with open(os.path.join(pred_path, fn), 'r') as pfile:
                pred_docs[fn] = pfile.read()

    true_docs = {}
    for fn in true_filenames:
        if fn[-3:] == "ann":
            with open(os.path.join(true_path, fn), 'r') as tfile:
                true_docs[fn] = tfile.read()

    pred_doc_ne = {}
    pred_doc_rl = {}
    for pred_file in pred_docs.keys():
        pred_ne = []
        pred_doc = pred_docs[pred_file]
        for line in pred_doc.split("\n"):
            if len(line) > 0 and line[0] == "T":
                ne = NameEntity()
                ne.init_with_str(line)
                pred_ne.append(ne)
        pred_doc_ne[pred_file] = pred_ne
    for pred_file in pred_docs.keys():
        pred_rl = []
        pred_doc = pred_docs[pred_file]
        pred_ne = pred_doc_ne[pred_file]
        for line in pred_doc.split("\n"):
            if len(line) > 0 and line[0] == "R":
                rl_id, type, arg1_id, arg2_id = process_relation_str(line)
                arg1 = get_name_entity(arg1_id, pred_ne)
                arg2 = get_name_entity(arg2_id, pred_ne)
                if arg1 is not None and arg2 is not None:
                    rl = Relation(rl_id, arg1, arg2, type)
                    pred_rl.append(rl)
        pred_doc_rl[pred_file] = pred_rl

    true_doc_ne = {}
    true_doc_rl = {}
    for true_file in true_docs.keys():
        true_ne = []
        true_doc = true_docs[true_file]
        for line in true_doc.split("\n"):
            if len(line) > 0 and line[0] == "T":
                ne = NameEntity()
                ne.init_with_str(line)
                true_ne.append(ne)
        true_doc_ne[true_file] = true_ne
    for true_file in true_docs.keys():
        true_rl = []
        true_doc = true_docs[true_file]
        true_ne = true_doc_ne[true_file]
        for line in true_doc.split("\n"):
            if len(line) > 0 and line[0] == "R":
                rl_id, type, arg1_id, arg2_id = process_relation_str(line)
                arg1 = get_name_entity(arg1_id, true_ne)
                arg2 = get_name_entity(arg2_id, true_ne)
                if type == 'has_title' or type == 'has_role':
                    type = HAS_TOR
                if arg1 is not None and arg2 is not None:
                    rl = Relation(rl_id, arg1, arg2, type)
                    true_rl.append(rl)
        true_doc_rl[true_file] = true_rl

    pred_count = 0
    correct_count = 0
    wrong_span_count = 0
    wrong_type_count = 0
    similar_count = 0
    for file in pred_doc_rl:
        print("============================================================")
        print(file)
        if file in true_doc_rl:
            pred_rl = pred_doc_rl[file]
            true_rl = true_doc_rl[file]
            for prl in pred_rl:
                pred_count += 1
                for trl in true_rl:

                    if is_overlap(prl.arg2.span, trl.arg2.span):
                        if not is_overlap(prl.arg1.span, trl.arg1.span):
                            print_prl_trl(prl, trl)
                        else:
                            correct_count += 1
        print("\n\n\n")

    # print("Wrong span count: ", wrong_span_count) # Name correct buy span wrong
    # print("Wrong type count: ", wrong_type_count) # Name correct buy type wrong
    # print("Similar count: ", similar_count)
    precision = correct_count*1./pred_count
    print("Precision:", precision)
    print("\t", correct_count, "/", pred_count)

    print("\n\n\n")
    print("\n\n\n")

    true_count = 0
    correct_count_true = 0
    for file in true_doc_rl:
        print("============================================================")
        print(file)
        if file in pred_doc_rl:
            true_rl = true_doc_rl[file]
            pred_rl = pred_doc_rl[file]
            for trl in true_rl:
                true_count += 1
                for prl in pred_rl:
                    # if trl.arg1.name[-1] in ",.;":
                    #     trl.arg1.name = trl.arg1.name[:-1]
                    #     trl.arg1.span[1] -= 1
                    # if prl.arg1.name[-1] in ",.;":
                    #     prl.arg1.name = prl.arg1.name[:-1]
                    #     prl.arg1.span[1] -= 1


                    # print(prl.arg1.name, "||||", trl.arg1.name)
                    # print(prl.arg2.name, "||||", trl.arg2.name)
                    # print("-----------------------------------")
                    # print(prl)
                    # print(trl)
                    # print(prl.arg1.span, "||||", trl.arg1.span)
                    # print(prl.arg2.span, "||||", trl.arg2.span)

                    if is_overlap(prl.arg2.span, trl.arg2.span):
                        if is_overlap(prl.arg1.span, trl.arg1.span):
                            correct_count_true += 1
                        else:
                            print_prl_trl(prl, trl)
        print("\n\n\n")


    recall = correct_count_true*1./true_count
    print("Recall:", recall)
    print("\t", correct_count_true, "/", true_count)
    print("F1 score:", 2. / (1./precision) + 1./recall)
