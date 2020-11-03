import os

ann_path = "ann/truth"
output_dir = "true_ne"

if __name__ == "__main__":
    pred_filenames = os.listdir(ann_path)

    pred_docs = {}
    for fn in pred_filenames:
        with open(os.path.join(ann_path, fn), 'r') as pfile:
            pred_docs[fn] = pfile.read()

    for pred_file in pred_docs.keys():
        ne_only = ""
        pred_doc = pred_docs[pred_file]
        for line in pred_doc.split("\n"):
            if len(line) > 0 and line[0] == "T":
                ne_only += line + "\n"
        outfile = open(os.path.join(output_dir, pred_file), 'w')
        outfile.write(ne_only)
        outfile.close()
