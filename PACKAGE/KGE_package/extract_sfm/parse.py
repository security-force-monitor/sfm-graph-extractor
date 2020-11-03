import os


def get_node(parse_tree, node_id):
    return parse_tree[node_id - 1]

def get_node_ids(parse_tree, word_position):
    cur_position = 0
    node_ids = []
    for node in parse_tree:
        if word_position[0]-1 <= cur_position and \
            cur_position + len(node[1]) <= word_position[1]+1:
            node_ids.append(node[0])
        cur_position += len(node[1]) + 1
    return node_ids

def path_to_root(parse_tree, leaf_id):
    path = [leaf_id]
    while path[-1] != 0:
        path.append(get_node(parse_tree, path[-1])[2])
    return path

def get_node_distance(parse_tree, source_id, target_id):
    source_path = path_to_root(parse_tree, source_id)
    target_path = path_to_root(parse_tree, target_id)
    common_ancestor = 0
    for sid in source_path:
        if sid in target_path:
            common_ancestor = sid
            break
    source_dist = source_path.index(common_ancestor)
    target_dist = target_path.index(common_ancestor)
    return source_dist + target_dist

def get_entity_distance(parse_tree, source_position, target_position):
    """ All functions above are helper functions for this one.

    position is list of length 2: [entity start position, entity end position]

    Calculate entity distance based on parse tree.
    """
    source_ids = get_node_ids(parse_tree, source_position)
    target_ids = get_node_ids(parse_tree, target_position)
    all_dists = []
    for sid in source_ids:
        for tid in target_ids:
            all_dists.append(get_node_distance(parse_tree, sid, tid))
    if len(all_dists) == 0:
        return None
    else:
        return min(all_dists)

def get_parse_tree(buffer_path, line_count):
    parse_path = os.path.join(buffer_path, str(line_count) + '.txt.conllu.pred')
    with open(parse_path) as parse_file:
        parse_tree = []
        for line in parse_file:
            if line.strip() != '':
                line_split = line.strip().split("\t")
                # [node_id, token, parent_id, dep_type]
                parse_tree.append([int(line_split[0]), line_split[1], int(line_split[6]), line_split[7]])
    return parse_tree


if __name__ == "__main__":
    with open("results/mytest.txt.conllu") as parse_file:
        parse_tree = []
        for line in parse_file:
            if line.strip() != '':
                line_split = line.strip().split()
                # [node_id, token, parent_id, dep_type]
                parse_tree.append([int(line_split[0]), line_split[1], int(line_split[6]), line_split[7]])
