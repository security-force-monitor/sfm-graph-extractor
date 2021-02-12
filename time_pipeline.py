import sys, os, subprocess
import time

NER_dir = "NER_v2"
RE_dir = "RE"
python_path = "python"
PIPELINE_PARTS = [True, True, True, False] # Whether to run a part of the pipeline: [NER, DEP_PARSE, RE, CLEAN_UP]
TIMER_FILE = 'time.txt'

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

    line_count = 0
    for id in input_ids:
        with open(os.path.join(input_dir, id + '.txt')) as input_file:
            for line in input_file:
                if len(line) > 1:
                    line_count += 1

    time_file = open(TIMER_FILE, 'w')
    time_file.write('There are a total of {} lines in {}\n'.format(line_count, input_dir))


    # ============================== Name Entity Recognition ==============================
    if PIPELINE_PARTS[0]:
        start_time = time.time()

        ner_path = os.path.join(cwd, NER_dir)
        os.chdir(ner_path)
        for id in input_ids:
            txt_path = os.path.join(input_dir, id + ".txt")
            # os.system("python " + "ner.py " + txt_path)
            subprocess.call([python_path, "ner.py", txt_path])

        total_time = time.time() - start_time
        averge_time = total_time / line_count
        time_file.write("NER took {} seconds to finish: \n\tAverge processing time for one line is {} seconds\n".format(total_time, averge_time))

    # ============================== Dependency Parsing ==============================

    if PIPELINE_PARTS[1]:
        start_time = time.time()

        os.chdir(os.path.join(cwd, RE_dir, "jPTDP"))
        # python_bin = os.path.join(cwd, RE_dir, "jPTDP-master/.DyNet/bin/python")
        script_file = os.path.join(cwd, RE_dir, "jPTDP/fast_parse.py")
        # converter_file = os.path.join(cwd, RE_dir, "jPTDP-master/utils/converter.py")
        # jPTDP_model_path = os.path.join(cwd, RE_dir, "jPTDP-master/sample/model256")
        # jPTDP_params_path = os.path.join(cwd, RE_dir, "jPTDP-master/sample/model256.params")
        for id in input_ids:
            # line_count = 0
            subprocess.call([python_path, script_file, os.path.join(input_dir, id + ".txt")])
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

        total_time = time.time() - start_time
        averge_time = total_time / line_count
        time_file.write("DEP_PARSE took {} seconds to finish: \n\tAverge processing time for one line is {} seconds\n".format(total_time, averge_time))


    # ============================== Relation Extraction ==============================
    if PIPELINE_PARTS[2]:
        start_time = time.time()

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
            subprocess.call([python_path, "relation_dep.py", parse_dir, os.path.join(input_dir, SDP_dir, id+".txt"), os.path.join(input_dir, SDP_dir, id+".ann")])

        total_time = time.time() - start_time
        averge_time = total_time / line_count
        time_file.write("{} took {} seconds to finish: \n\tAverge processing time for one line is {} seconds\n".format(SDP_dir, total_time, averge_time))


        start_time = time.time()

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
        subprocess.call([python_path, "relation_nn.py", os.path.join(input_dir, NN_dir)])

        total_time = time.time() - start_time
        averge_time = total_time / line_count
        time_file.write("{} took {} seconds to finish: \n\tAverge processing time for one line is {} seconds\n".format(NN_dir, total_time, averge_time))

    time_file.close()


    # ============================== Clean Up ==============================
    if PIPELINE_PARTS[3]:
        for id in input_ids:
            subprocess.call(["rm", os.path.join(input_dir, id + ".ann")])
            subprocess.call(["rm", os.path.join(input_dir, id + ".txt")])
            subprocess.call(["rm", "-r", os.path.join(input_dir, id)])
            subprocess.call(["rm", "-r", os.path.join(input_dir, NN_dir, id)])
