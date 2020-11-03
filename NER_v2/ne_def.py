HAS_RANK = 'has_rank'
HAS_TOR = 'has_title_or_role'
HAS_TITLE = 'has_title'
HAS_ROLE = 'has_role'
IS_POSTED = 'is_posted'
all_relation_types = [HAS_RANK, HAS_TOR, IS_POSTED, HAS_TITLE, HAS_ROLE]

label_mapping = {'Person':       'PER',
                 'Rank':         'RNK',
                 'Organization': 'ORG',
                 'Title':        'TOR',
                 'Role':         'TOR',
                 'Title_Role':   'TOR',
                 'Location':     'LOC',
                 'Misclass':     'MISC'}

inv_label_mapping = {v: k for k, v in label_mapping.items()}
inv_label_mapping['TOR'] = 'Title_Role'

class NameEntity():
    def __init__(self, id = -1, name = None, type = None, span = [-1, -1]):
        self.id = id # int
        self.name = name # str
        self.type = type # str, use label_mapping
        self.span = span # list, len == 2: [int, int]

    def init_with_str(self, line):
        if line[0] == 'T':
            line_split = line.strip().split()
            self.id = int(line_split[0][1:])
            self.name = ' '.join(line_split[4:])
            self.type = label_mapping[line_split[1]]
            self.span = [ int(line_split[2]), int(line_split[3]) ]
        else:
            raise NameError(line, 'is not a name entity')

    def __str__(self):
        return str(self.id) + " |||| " + self.name + " |||| " + inv_label_mapping[self.type] + " |||| " + str(self.span)

    def get_ann_str(self):
        return "T{}	{} {} {}	{}".format(self.id, \
            inv_label_mapping[self.type], self.span[0], self.span[1], self.name)


class Relation():
    def __init__(self, id, arg1, arg2, type):
        if type in all_relation_types:
            self.type = type
        else:
            raise NameError(type, ': such relationship does not exist!')
        self.id = id # int
        self.arg1 = arg1 # class NameEntity
        self.arg2 = arg2 # class NameEntity

    def __str__(self):
        return str(self.id) + ': ' + self.arg1.name + ' --- [' + self.type + '] ---> ' + self.arg2.name

    def get_ann_str(self):
        return "R{}	{} Arg1:T{} Arg2:T{}".format(self.id, \
            self.type, self.arg1.id, self.arg2.id)


def correct_position(whole_str, query_idx, query_str):
    offset = 0
    left_idx = query_idx - offset
    right_idx = query_idx + offset
    while left_idx >= 0 or right_idx < len(whole_str):
        left_idx = query_idx - offset
        if left_idx >= 0 and whole_str[left_idx: left_idx + len(query_str)] == query_str:
            return (left_idx, left_idx + len(query_str))

        right_idx = query_idx + offset
        if right_idx < len(whole_str) and whole_str[right_idx: right_idx + len(query_str)] == query_str:
            return (right_idx, right_idx + len(query_str))

        offset += 1

    return None, None
