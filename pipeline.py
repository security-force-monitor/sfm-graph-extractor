import sys, os, subprocess
sys.path.insert(1, 'RE/utils')
from parse_script import *

NER_dir = "NER_v2"
RE_dir = "RE"
PIPELINE_PARTS = [True, True, True, False] # Whether to run a part of the pipeline: [NER, DEP_PARSE, RE, CLEAN_UP]

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Type in the path to the input directory (use absolute path and cannot include [space]).")
        exit()

    cwd = os.getcwd()
    input_dir = sys.argv[1]

    input_files = os.listdir(input_dir)
    input_ids = []
    for input_file in input_files:
        if input_file[-3:] == "txt":
            input_ids.append(input_file[:-4])

    # ============================== Name Entity Recognition ==============================
    if PIPELINE_PARTS[0]:
        ner_path = os.path.join(cwd, NER_dir)
        os.chdir(ner_path)
        for id in input_ids:
            txt_path = os.path.join(input_dir, id + ".txt")
            # os.system("python " + "ner.py " + txt_path)
            subprocess.call(["python", "ner.py", txt_path])

    # ============================== Dependency Parsing ==============================
    if PIPELINE_PARTS[1]:
        os.chdir(os.path.join(cwd, RE_dir, "jPTDP-master"))
        python_bin = os.path.join(cwd, RE_dir, "jPTDP-master/.DyNet/bin/python")
        script_file = os.path.join(cwd, RE_dir, "jPTDP-master/fast_parse.py")
        # converter_file = os.path.join(cwd, RE_dir, "jPTDP-master/utils/converter.py")
        # jPTDP_model_path = os.path.join(cwd, RE_dir, "jPTDP-master/sample/model256")
        # jPTDP_params_path = os.path.join(cwd, RE_dir, "jPTDP-master/sample/model256.params")
        for id in input_ids:
            line_count = 0
            subprocess.call([python_bin, script_file, os.path.join(input_dir, id + ".txt")])
            # with open(os.path.join(input_dir, id + ".txt"), 'r') as input_file:
            #     # os.system("mkdir " + os.path.join(input_dir, id))
            #     subprocess.call(["mkdir", os.path.join(input_dir, id)])
            #     for line in input_file:
            #         line_strip = clean_str(line.strip())
            #         if line_strip != "" and not line_strip.isspace():
            #             line_path = os.path.join(input_dir, id, str(line_count) + ".txt")
            #             with open(line_path, 'w') as line_file:
            #                 line_file.write(line_strip)
            #
            #             parse_input = os.path.join(input_dir, id, str(line_count) + ".txt.conllu")
            #             parse_output = parse_input + ".pred"
            #
            #             subprocess.call([python_bin, converter_file, line_path])
            #             subprocess.call([python_bin, script_file, "--predict", \
            #                 "--model", jPTDP_model_path, "--params", jPTDP_params_path, \
            #                 "--test", parse_input, "--outdir", input_dir, "--output", parse_output])
            #
            #             line_count += 1

    if PIPELINE_PARTS[2]:
        SDP_dir = "SDP"
        subprocess.call(["mkdir", os.path.join(input_dir, SDP_dir)])
        # subprocess.call(["cp", os.path.join(input_dir, "*.txt"), os.path.join(input_dir, SDP_dir)])
        # subprocess.call(["cp", os.path.join(input_dir, "*.ann"), os.path.join(input_dir, SDP_dir)])
        os.chdir(os.path.join(cwd, RE_dir, "2. dep"))
        for id in input_ids:
            parse_dir = os.path.join(input_dir, id)
            subprocess.call(["cp", parse_dir + ".txt", os.path.join(input_dir, SDP_dir)])
            subprocess.call(["cp", parse_dir + ".ann", os.path.join(input_dir, SDP_dir)])
            # subprocess.call(["cp", "-r", parse_dir, os.path.join(input_dir, SDP_dir)])
            subprocess.call(["python", "relation_dep.py", parse_dir, os.path.join(input_dir, SDP_dir, id+".txt"), os.path.join(input_dir, SDP_dir, id+".ann")])


        NN_dir = "NN"
        subprocess.call(["mkdir", os.path.join(input_dir, NN_dir)])
        # subprocess.call(["cp", os.path.join(input_dir, "*.txt"), os.path.join(input_dir, SDP_dir)])
        # subprocess.call(["cp", os.path.join(input_dir, "*.ann"), os.path.join(input_dir, SDP_dir)])
        os.chdir(os.path.join(cwd, RE_dir, "3. nn"))
        for id in input_ids:
            parse_dir = os.path.join(input_dir, id)
            subprocess.call(["cp", parse_dir + ".txt", os.path.join(input_dir, NN_dir)])
            subprocess.call(["cp", parse_dir + ".ann", os.path.join(input_dir, NN_dir)])
            subprocess.call(["cp", "-r", parse_dir, os.path.join(input_dir, NN_dir)])
        #     subprocess.call(["cp", parse_dir + ".ann", parse_dir + ".ann.nn"])
        subprocess.call(["python", "relation_nn.py", os.path.join(input_dir, NN_dir)])

    if PIPELINE_PARTS[3]:
        for id in input_ids:
            subprocess.call(["rm", os.path.join(input_dir, id + ".ann")])
            subprocess.call(["rm", os.path.join(input_dir, id + ".txt")])
            subprocess.call(["rm", "-r", os.path.join(input_dir, id)])
            subprocess.call(["rm", "-r", os.path.join(input_dir, NN_dir, id)])
