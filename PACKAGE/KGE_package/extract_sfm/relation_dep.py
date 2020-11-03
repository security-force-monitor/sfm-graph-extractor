import os, sys
from ne_def import *
import parse


def get_name_entity(ne_list, id):
    for ne in ne_list:
        if id == ne.id:
            return ne
    return None

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

    if len(sys.argv) < 2:
        print("Enter path to buffer for .conllu files, raw text file and annotation file")
        exit()
    elif len(sys.argv) < 3:
        print(".conllu file buffer: " + sys.argv[1])
        print("Enter the raw text file and the annotation file")
        exit()
    elif len(sys.argv) < 4:
        print(".conllu file buffer: " + sys.argv[1])
        print("Text from file: " + sys.argv[2])
        print("Enter the annotation file")
        exit()
    else:
        print(".conllu file buffer: " + sys.argv[1])
        print("Text from file: " + sys.argv[2])
        print("Annotation from file: " + sys.argv[3])

    buffer_path = sys.argv[1]
    txt_path = sys.argv[2]
    ann_path = sys.argv[3]

    with open(ann_path) as ann_file:
        all_persons = []
        all_others = []
        for line in ann_file:
            new_ne = NameEntity()
            new_ne.init_with_str(line)
            if new_ne.type == "PER":
                all_persons.append(new_ne)
            else:
                all_others.append(new_ne)

    all_lines = []
    with open(txt_path) as input_file:
        for line in input_file:
            # print(line.strip())
            line_strip = line.strip()
            if line_strip != "" and not line_strip.isspace():
                all_lines.append(line)

    with open(txt_path) as input_file:
        whole_doc = input_file.read()

    line_count = 0
    rel_count = 0
    sentence_start = 0
    while os.path.isfile( os.path.join(buffer_path, str(line_count) + '.txt.conllu.pred') ):
        sentence_start, sentence_end = correct_position(whole_doc, sentence_start, all_lines[line_count])

        parse_tree = parse.get_parse_tree(buffer_path, line_count)
        person_nodes = get_person_nodes(parse_tree, all_persons, [sentence_start, sentence_end])

        if not bool(person_nodes):
            sentence_start += len(all_lines[line_count]) + 1
            line_count += 1
            continue

        relations = []
        for ne in all_others:
            if ne.span[1] < sentence_start or ne.span[0] > sentence_end:
                continue
            else:
                ne_sentence_span = [ne.span[0] - sentence_start, ne.span[1] - sentence_start]
            min_dist = float('inf')
            best_per = None

            left_person = None
            right_person = None
            for per in all_persons:
                if per.span[0] >= sentence_start and per.span[1] <= sentence_end:
                    if per.span[1] <= ne.span[0]:
                        if left_person is None or per.span[1] > left_person.span[1]:
                            left_person = per
                    if per.span[0] >= ne.span[1]:
                        if right_person is None or per.span[0] < right_person.span[0]:
                            right_person = per

            nearest_persons = []

            stn_split_punc = ";"
            if left_person is not None:
                txt_between_span = [left_person.span[1] - sentence_start, ne.span[0] - sentence_start]
                txt_between = all_lines[line_count][txt_between_span[0]: txt_between_span[1]]
                if not is_diff_stn(stn_split_punc, txt_between):
                    nearest_persons.append(left_person)
            if right_person is not None:
                txt_between_span = [ne.span[1] - sentence_start, right_person.span[0] - sentence_start]
                txt_between = all_lines[line_count][txt_between_span[0]: txt_between_span[1]]
                if not is_diff_stn(stn_split_punc, txt_between):
                    nearest_persons.append(right_person)

            for per_idx, per in enumerate(nearest_persons):
                per_sentence_span = [per.span[0] - sentence_start, per.span[1] - sentence_start]
                cur_dist = parse.get_entity_distance(parse_tree, ne_sentence_span, per_sentence_span)
                if cur_dist is not None and cur_dist < min_dist:
                    min_dist = cur_dist
                    best_per = per

            if best_per is not None:
                if ne.type == "TOR":
                    rel_count += 1
                    relations.append(Relation(rel_count, best_per, ne, HAS_TOR))
                elif ne.type == "RNK":
                    rel_count += 1
                    relations.append(Relation(rel_count, best_per, ne, HAS_RANK))
                elif ne.type == "ORG":
                    rel_count += 1
                    relations.append(Relation(rel_count, best_per, ne, IS_POSTED))


        ann_file = open(ann_path, 'a')
        for rl in relations:
            ann_file.write(rl.get_ann_str() + '\n')
        ann_file.close()

        sentence_start += len(all_lines[line_count]) + 1
        line_count += 1
