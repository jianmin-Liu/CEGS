import os
import shutil

from pybatfish.client.session import Session
from utils import *
from generator import get_sub_topology, convert_configuration_to_json
from utils import json_to_config

def get_feedback(bf, errors):
    incorrect_nodes = []
    error_lines = {}
    for index, row in errors.iterrows():
        node_name = row['File_Name'].split('/')[-1].split('.')[0]
        incorrect_nodes.append(node_name)
        error_lines[node_name] = []
    warning = bf.q.parseWarning().answer().frame()

    for index, row in warning.iterrows():
        node = row['Filename'].split('/')[-1].split('.')[0]
        if node in incorrect_nodes:
            comment = row['Comment']
            if 'not currently supported' not in comment:
                error_lines[node].append(row['Line'])

    feedback = ""
    for node, lines in error_lines.items():
        lines = sorted(map(str, lines))
        if len(lines) == 1:
            lines_str = lines[0]
        elif len(lines) == 2:
            lines_str = " and ".join(lines)
        else:
            lines_str = ", ".join(lines[:-1]) + ", and " + lines[-1]

        feedback += f"The configuration of {node} has syntax errors in lines {lines_str}. "

    return feedback, incorrect_nodes


def correct_configs(nodes, configs, topology, feedback):
    with open('./prompts/reflection/prefix.txt', 'r') as f:
        prompt_prefix = f.read()

    with open('./prompts/reflection/syntax_suffix.txt', 'r') as f:
        suffix_template = f.read()

    suffix_prompt = suffix_template.format(node_set=nodes)

    prompt = prompt_prefix + f'\n\nNetwork topology: {topology}\n\n' + f'Configurations:\n{configs}\n\n' + f'Feedback:\n{feedback}\n\n' + suffix_prompt
    #print('reflect_prompt', prompt)
    reflection = ask_LLM(prompt)
    #print('reflection', reflection)
    return reflection


def verify_syntax(configurations, topology):
    if type(configurations) == str:
        configurations = eval(configurations)
    loops = 0
    new_configurations = {}
    while True:
        if len(new_configurations.keys()) > 0:
            for node in new_configurations.keys():
                configurations[node] = new_configurations[node]

        folder_path = 'batfish'
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
        os.mkdir(folder_path)
        os.mkdir(f'{folder_path}/configs')
        for node, configuration in configurations.items():
            with open(f'{folder_path}/configs/{node}.cfg', 'w') as file:
                file.write(configuration)

        bf = Session(host="localhost")
        NETWORK_NAME = "network"
        SNAPSHOT_NAME = "snapshot"

        SNAPSHOT_DIR = './batfish'

        bf.set_network(NETWORK_NAME)
        bf.init_snapshot(SNAPSHOT_DIR, name=SNAPSHOT_NAME, overwrite=True)

        parse_status = bf.q.fileParseStatus().answer().frame()
        errors = parse_status[parse_status['Status'] != 'PASSED']

        if not errors.empty:
            feedback, incorrect_nodes = get_feedback(bf, errors)
            loops += 1
            subtopo = get_sub_topology(topology, incorrect_nodes)
            incorrect_configs = {}
            for node in incorrect_nodes:
                incorrect_configs[node] = configurations[node]

            configs_str = json_to_config(json.dumps(incorrect_configs))
            new_output = correct_configs(incorrect_nodes, configs_str, subtopo, feedback)
            new_configurations = eval(convert_configuration_to_json(new_output))
        else:
            break
    return loops, configurations

