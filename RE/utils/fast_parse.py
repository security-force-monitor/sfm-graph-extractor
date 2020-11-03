import os, subprocess
import sys
import re
from optparse import OptionParser
import pickle, utils, learner, os, os.path, time

params = "sample/model256.params"
model = "sample/model256"


# remove annoying characters
# https://stackoverflow.com/questions/6609895/efficiently-replace-bad-characters
chars = {
    '\xc2\x82' : ',',        # High code comma
    '\xc2\x84' : ',,',       # High code double comma
    '\xc2\x85' : '...',      # Tripple dot
    '\xc2\x88' : '^',        # High carat
    '\xc2\x91' : '\x27',     # Forward single quote
    '\xc2\x92' : '\x27',     # Reverse single quote
    '\xc2\x93' : '\x22',     # Forward double quote
    '\xc2\x94' : '\x22',     # Reverse double quote
    '\xc2\x95' : ' ',
    '\xc2\x96' : '-',        # High hyphen
    '\xc2\x97' : '--',       # Double hyphen
    '\xc2\x99' : ' ',
    '\xc2\xa0' : ' ',
    '\xc2\xa6' : '|',        # Split vertical bar
    '\xc2\xab' : '<<',       # Double less than
    '\xc2\xbb' : '>>',       # Double greater than
    '\xc2\xbc' : '1/4',      # one quarter
    '\xc2\xbd' : '1/2',      # one half
    '\xc2\xbe' : '3/4',      # three quarters
    '\xca\xbf' : '\x27',     # c-single quote
    '\xcc\xa8' : '',         # modifier - under curve
    '\xcc\xb1' : ''          # modifier - under line
}

def replace_chars(match):
    char = match.group(0)
    return chars[char]

def clean_str(text):
    return re.sub('(' + '|'.join(chars.keys()) + ')', replace_chars, text)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Enter text filename")
        exit()
    else:
        print("Text from file: " + sys.argv[1])

    buffer = sys.argv[1][:-4]
    os.system("mkdir " + buffer)

    # Load model
    with open(params, 'r') as paramsfp:
        words, w2i, c2i, pos, rels, stored_opt = pickle.load(paramsfp)
    stored_opt.external_embedding = None
    parser = learner.jPosDepLearner(words, pos, rels, w2i, c2i, stored_opt)
    parser.Load(model)
    # print('Pre-trained model loaded')

    # Create one file for each line
    input_path = sys.argv[1]
    cwd = os.getcwd()
    line_count = 0
    with open(input_path) as input_file:
        for line in input_file:
            line_strip = clean_str(line.strip())
            if line_strip != "" and not line_strip.isspace():
                line_file = open(os.path.join(buffer, str(line_count) + ".txt"), 'w')
                line_file.write(line_strip)
                line_file.close()
                line_count += 1

    for line_idx in range(line_count):
        os.system("python utils/converter.py " + os.path.join(buffer, str(line_idx) + ".txt"))

        output = buffer
        conll_test = os.path.join(output, str(line_idx) + ".txt.conllu")
        conll_test_output = str(line_idx) + ".txt.conllu.pred"
        testoutpath = os.path.join(output, conll_test_output)
        # print 'Predicting POS tags and parsing dependencies'
        with open(testoutpath, 'w') as fh:
            for sentence in parser.Predict(conll_test):
                for entry in sentence[1:]:
                    fh.write(str(entry) + '\n')
                fh.write('\n')
        print "Finished Line - " + str(line_idx)

        #
        # os.system("python jPTDP.py --predict --model sample/model256 --params sample/model256.params --test buffer/" + str(line_idx) + ".txt.conllu --outdir buffer/ --output " + str(line_idx) + ".txt.conllu.pred")
