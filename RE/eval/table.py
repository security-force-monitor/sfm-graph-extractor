import os
import sys
import csv
sys.path.insert(1, '../3. nn')
from data import *
from path import *
from config import *

sys.path.insert(1, '../../NER_v2')
from ne_def import *
from eval import *



pred_path = "../3. nn/out"
# pred_path = "dep"
true_path = "ann/truth"
csv_filename = "table.csv"

GITHUB_LINK = 'https://github.com/security-force-monitor/nlp_starter_dataset/blob/master/annotated_sources/'
LINK_SUFFIX = ".txt"

def find_person_rl(per_ne, rl_list):
    per_rl = []
    for rl in rl_list:
        if rl.arg1.id == per_ne.id:
            per_rl.append(rl)
    return per_rl

def get_ne_within_sentence(ne_list, sentence_start, sentence_end):
    result_ne = []
    for ne in ne_list:
        if sentence_start <= ne.span[0] and ne.span[1] <= sentence_end:
            result_ne.append(ne)
    return result_ne

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


    pred_doc_ids = []
    for fn in pred_filenames:
        if fn[-3:] == "ann":
            with open(os.path.join(pred_path, fn), 'r') as pfile:
                # pred_docs[fn] = pfile.read()
                pred_doc_ids.append(fn[:-4])


    table = []
    for doc_id in pred_doc_ids:
        # print("========================================\n" + doc_id)

        txt_path = os.path.join(pred_path, doc_id + ".txt")
        buffer_path = os.path.join(pred_path, doc_id)

        all_lines = []
        with open(txt_path) as input_file:
            for line in input_file:
                # print(line.strip())
                line_strip = line.strip()
                if line_strip != "" and not line_strip.isspace():
                    all_lines.append(line)

        with open(txt_path) as input_file:
            whole_doc = input_file.read()

        # all_persons, all_others, true_relations = get_ne_rl(ann_path)

        line_count = 0
        rel_count = 0
        sentence_start = 0
        while os.path.isfile( os.path.join(buffer_path, str(line_count) + '.txt.conllu.pred') ):
            # print('-----------------Line: ', line_count + 1)
            if line_count >= len(all_lines):
                break
            sentence_start, sentence_end = correct_position(whole_doc, sentence_start, all_lines[line_count])

            pred_stn_nes = get_ne_within_sentence(get_ne_of_type(pred_doc_ne[doc_id + '.ann'] , "PER"), sentence_start, sentence_end)
            true_stn_nes = get_ne_within_sentence(get_ne_of_type(true_doc_ne[doc_id + '.ann'] , "PER"), sentence_start, sentence_end)

            for pred_per in pred_stn_nes:
                table_row = {}
                table_row["pred_person"] = pred_per.name
                table_row["pred_rank"] = []
                table_row["pred_title_or_role"] = []
                table_row["pred_organization"] = []
                table_row["true_person"] = None
                table_row["true_rank"] = []
                table_row["true_title_or_role"] = []
                table_row["true_organization"] = []
                table_row["line_number"] = line_count + 1
                table_row["source_uuid"] = doc_id
                table_row["link_to_ann"] = GITHUB_LINK + doc_id + LINK_SUFFIX

                per_rls = find_person_rl(pred_per, pred_doc_rl[doc_id + '.ann'])
                for rl in per_rls:
                    if rl.arg2.type == 'RNK':
                        table_row["pred_rank"].append(rl.arg2.name)
                    elif rl.arg2.type == 'TOR':
                        table_row["pred_title_or_role"].append(rl.arg2.name)
                    elif rl.arg2.type == 'ORG':
                        table_row["pred_organization"].append(rl.arg2.name)

                for true_per in true_stn_nes:
                    if is_overlap(pred_per.span, true_per.span):
                        table_row["true_person"] = true_per.name

                        per_rls = find_person_rl(true_per, true_doc_rl[doc_id + '.ann'])
                        for rl in per_rls:
                            if rl.arg2.type == 'RNK':
                                table_row["true_rank"].append(rl.arg2.name)
                            elif rl.arg2.type == 'TOR':
                                table_row["true_title_or_role"].append(rl.arg2.name)
                            elif rl.arg2.type == 'ORG':
                                table_row["true_organization"].append(rl.arg2.name)

                        true_stn_nes.remove(true_per)
                table.append(table_row)


            # Missed true persons
            for true_per in true_stn_nes:
                table_row = {}
                table_row["pred_person"] = None
                table_row["pred_rank"] = []
                table_row["pred_title_or_role"] = []
                table_row["pred_organization"] = []
                table_row["true_person"] = true_per.name
                table_row["true_rank"] = []
                table_row["true_title_or_role"] = []
                table_row["true_organization"] = []
                table_row["line_number"] = line_count + 1
                table_row["source_uuid"] = doc_id
                table_row["link_to_ann"] = GITHUB_LINK + doc_id + LINK_SUFFIX

                per_rls = find_person_rl(true_per, true_doc_rl[doc_id + '.ann'])
                for rl in per_rls:
                    if rl.arg2.type == 'RNK':
                        table_row["true_rank"].append(rl.arg2.name)
                    elif rl.arg2.type == 'TOR':
                        table_row["true_title_or_role"].append(rl.arg2.name)
                    elif rl.arg2.type == 'ORG':
                        table_row["true_organization"].append(rl.arg2.name)

                table.append(table_row)

                # pp.pprint(table_row)


            sentence_start += len(all_lines[line_count]) + 1
            line_count += 1

    table_columns = ['pred_person', 'true_person', 'pred_rank', 'true_rank', \
        'pred_title_or_role', 'true_title_or_role', 'pred_organization', 'true_organization', \
        'line_number', 'source_uuid', 'link_to_ann']
    # with open(csv_filename, 'w', newline='') as csvfile:
    #     spamwriter = csv.writer(csvfile, delimiter='#',
    #                             quotechar='|', quoting=csv.QUOTE_MINIMAL)
    #     for row in table:
    #         row_list = []
    #         for col in table_columns:
    #             row_list.append(row[col])
    #
    #     spamwriter.writerow(['Spam'] * 5 + ['Baked Beans'])
    #     spamwriter.writerow(['Spam', 'Lovely Spam', 'Wonderful Spam'])

    with open(csv_filename, 'w', newline='') as outcsv:
        writer = csv.DictWriter(outcsv, fieldnames = table_columns)
        writer.writeheader()
        writer.writerows(row for row in table)
