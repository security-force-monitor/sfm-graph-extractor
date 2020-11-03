import os
import sys
from ne_def import *


def get_name_entity(id, ne_list):
    for ne in ne_list:
        if ne.id == id:
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

def get_ne2node_mapping(parse_tree, ne_list, word_start, span_tolerance = 0):
    non_persons = {}
    for idx in range(len(parse_tree)):
        word_end = word_start + len(parse_tree[idx][1])
        for ne in ne_list:
            if ne.span[0] - span_tolerance <= word_start and \
                    word_end<= ne.span[1] + span_tolerance :
                if ne.id not in non_persons:
                    non_persons[ne.id] = [parse_tree[idx][0]]
                else:
                    non_persons[ne.id].append(parse_tree[idx][0])
        word_start = word_end + 1
    return non_persons

def get_neighbors(parse_tree, elm):
    """ An elm has structure: [cur_node, path from from_node to cur_node] """
    neighbors = []
    cur_node = parse_tree[elm[0] - 1]
    if cur_node[2] != 0 and cur_node[2] not in elm[1] + [elm[0]]:
        parent_elm = [ cur_node[2], elm[1] + [elm[0]] ]
        neighbors.append(parent_elm)

    for node in parse_tree:
        if node[2] == elm[0] and node[0] not in elm[1] + [elm[0]]:
            new_elm = [node[0], elm[1] + [elm[0]]]
            neighbors.append(new_elm)

    return neighbors

def get_node_path(parse_tree, from_id, to_id):
    """ Find the path between nodes """
    queue = [ [from_id, []] ]
    while len(queue) != 0:
        cur_elm = queue.pop(0)
        if cur_elm[0] == to_id:
            return cur_elm
        neighbors = get_neighbors(parse_tree, cur_elm)
        queue += neighbors
    return None

def get_ne_path(parse_tree, from_ids, to_ids):
    """
    A name entity could span multiple nodes,
    we look for the closest path
    """
    all_paths = []
    min_dist = float('inf')
    min_path = None
    for from_id in from_ids:
        for to_id in to_ids:
            cur_path = get_node_path(parse_tree, from_id, to_id)
            cur_dist = len(cur_path[1])
            if cur_dist < min_dist:
                min_dist = cur_dist
                min_path = cur_path
    return min_path

def print_path(parse_tree, path):
    whole_path = path[1] + [path[0]]
    for idx in range(len(whole_path) - 1):
        left_node = parse_tree[whole_path[idx] - 1]
        right_node = parse_tree[whole_path[idx + 1] - 1]
        if left_node[2] == right_node[0]:
            print(" {}  <--({})---  ".format(left_node[1], left_node[3]), end = '')
        if left_node[0] == right_node[2]:
            print(" {}  ---({})-->  ".format(left_node[1], right_node[3]), end = '')
        if idx == len(whole_path) - 2:
            print(right_node[1])
    print('')

def get_type_path(parse_tree, path):
    type_path = []
    whole_path = path[1] + [path[0]]
    for idx in range(len(whole_path) - 1):
        left_node = parse_tree[whole_path[idx] - 1]
        right_node = parse_tree[whole_path[idx + 1] - 1]
        if left_node[2] == right_node[0]:
            type_path.append(left_node[3])
        if left_node[0] == right_node[2]:
            type_path.append(right_node[3])
    return tuple(type_path)

def get_ne_rl(ann_path):
    with open(ann_path) as ann_file:
        all_persons = []
        all_others = []
        for line in ann_file:
            if line[0] == 'T':
                new_ne = NameEntity()
                new_ne.init_with_str(line)
                if new_ne.type == "PER":
                    all_persons.append(new_ne)
                else:
                    all_others.append(new_ne)

    with open(ann_path) as ann_file:
        all_relations = []
        for line in ann_file:
            if line[0] == 'R':
                line_split = line.split()
                rl_id = int(line_split[0][1:])
                rl_type = line_split[1]
                arg1_id = int(line_split[2][6:])
                arg2_id = int(line_split[3][6:])
                arg1 = get_name_entity(arg1_id, all_persons + all_others)
                arg2 = get_name_entity(arg2_id, all_persons + all_others)
                new_rl = Relation(rl_id, arg1, arg2, rl_type)
                all_relations.append(new_rl)
    return all_persons, all_others, all_relations

def get_lines(txt_path):
    all_lines = []
    with open(txt_path) as input_file:
        for line in input_file:
            if line.strip() != "":
                all_lines.append(line)
    return all_lines


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

    all_persons, all_others, all_relations = get_ne_rl(ann_path)
    all_lines = get_lines(txt_path)

    line_count = 0
    rel_count = 0
    sentence_start = 0
    while os.path.isfile( os.path.join(buffer_path, str(line_count) + '.txt.conllu.pred') ):
        sentence_end = sentence_start + len(all_lines[line_count])
        print("=========================", line_count)
        # print("+++++ sentence : ", [sentence_start, sentence_end])

        parse_tree = parse.get_parse_tree(buffer_path, line_count)

        # ne id --> node id
        others_nodes = get_ne2node_mapping(parse_tree, all_others, sentence_start, 1)
        persons_nodes = get_ne2node_mapping(parse_tree, all_persons, sentence_start, 1)

        # import pdb; pdb.set_trace()
        for per_id in persons_nodes:
            for ot_id in others_nodes:
                for rl in all_relations:
                    if rl.arg1.id == per_id and rl.arg2.id == ot_id:
                        print('-------------------', rl)
                        path = get_ne_path(parse_tree, persons_nodes[per_id], others_nodes[ot_id])
                        print(get_type_path(parse_tree, path))
                        print_path(parse_tree, path)


        sentence_start += len(all_lines[line_count]) + 1
        line_count += 1
