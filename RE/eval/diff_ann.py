import os
import sys
sys.path.insert(1, '../../NER_v2')
from ne_def import *
sys.path.insert(1, '../utils')
from path import *
from eval import *

pred_path = "../3. nn/out"
# pred_path = "dep"
true_path = "truth"

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

    for file in pred_doc_rl:
        print("============================================================")
        print(file)
        if file in true_doc_rl:
            pred_rl = pred_doc_rl[file]
            true_rl = true_doc_rl[file]
            for prl in pred_rl:
                for trl in true_rl:
                    if is_overlap(prl.arg2.span, trl.arg2.span):
                        if not is_overlap(prl.arg1.span, trl.arg1.span):
                            print_prl_trl(prl, trl)
        print("\n\n\n")

    print("\n\n\n")
    print("\n\n\n")

    for file in true_doc_rl:
        print("============================================================")
        print(file)
        if file in pred_doc_rl:
            true_rl = true_doc_rl[file]
            pred_rl = pred_doc_rl[file]
            for trl in true_rl:
                for prl in pred_rl:
                    if is_overlap(prl.arg2.span, trl.arg2.span):
                        if not is_overlap(prl.arg1.span, trl.arg1.span):
                            print_prl_trl(prl, trl)
        print("\n\n\n")
