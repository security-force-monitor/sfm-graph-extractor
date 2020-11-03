import sys
sys.path.insert(1, '../../NER_v2')
from ne_def import *


def get_relations(rel_count, pred_entities):
    person_entities = []
    for entity in pred_entities:
        if entity.type == "PER":
            person_entities.append(entity)

    if len(person_entities) == 0:
        return rel_count, None

    relations = []
    for entity in pred_entities:
        idx = 0
        while idx < len(person_entities) - 1:
            if entity.span[0] >= person_entities[idx].span[1]:
                idx += 1
            else:
                break

        if entity.type == "TOR":
            relations.append(Relation(rel_count, person_entities[idx], entity, HAS_TOR))
            rel_count += 1
        elif entity.type == "RNK":
            relations.append(Relation(rel_count, person_entities[idx], entity, HAS_RANK))
            rel_count += 1
        elif entity.type == "ORG":
            relations.append(Relation(rel_count, person_entities[idx], entity, IS_POSTED))
            rel_count += 1
    # print(relations)
    if len(relations) == 0:
        return rel_count, None
    return rel_count, relations


if __name__ == '__main__':
    # From 0a33bba3-ef02-46e0-897c-0195d43ab626
    if len(sys.argv) < 2:
        print("Enter raw text filename and its annotation filename")
        exit()
    elif len(sys.argv) < 3:
        print("Text from file: " + sys.argv[1])
        print("Enter its annotation filename")
        exit()
    else:
        print("Text from file: " + sys.argv[1])
        print("Annotation from file: " + sys.argv[2])

    # Read files
    txt_path = sys.argv[1]
    with open(txt_path) as test_file:
        test_sentences = test_file.readlines()
        doc_sentences = [sentence.strip() for sentence in test_sentences if sentence.strip() != '']

    ann_path = sys.argv[2]
    with open(ann_path) as ann_file:
        all_entities = []
        for line in ann_file:
            new_ne = NameEntity()
            new_ne.init_with_str(line)
            all_entities.append(new_ne)

    # Nearest-Person relation extraction
    rel_count = 1
    cur_sentence_start = 0
    doc_relations = []
    for sentence in doc_sentences:
        pred_entities = []
        for en in all_entities:
            if cur_sentence_start <= en.span[0] and \
                    en.span[1] <= cur_sentence_start + len(sentence):
                pred_entities.append(en)

        rel_count, relations = get_relations(rel_count, pred_entities)
        if relations is not None:
            doc_relations += relations
        cur_sentence_start += len(sentence) + 2

    # Write relation annotations into .ann
    ann_path = sys.argv[1][:-4] + '.ann'
    ann_file = open(ann_path,'a')
    for r in doc_relations:
        ann_file.write(r.get_ann_str() + '\n')
    ann_file.close()
