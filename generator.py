import concurrent
import copy
import json
import math
import os
import re
import shutil
import time
from utils import *
from netcomplete.netcomplete_ebgp_eval import *
from netcomplete.ospf_eval import ospfeval


def get_node_information(topo, node_name):
    for element in topo['nodes']:
        if element['name'] == node_name:
            return element

def get_sub_topology(global_topology, sub_nodes):
    sub_topology = {'nodes': [], 'edges': []}

    neighbors = []
    for node in sub_nodes:
        for edge in global_topology['edges']:
            node1_name = edge["node1"]["name"]
            node2_name = edge["node2"]["name"]

            if node1_name == node:
                neighbors.append(node2_name)
                sub_topology['edges'].append(edge)
            elif node2_name == node:
                neighbors.append(node1_name)
                sub_topology['edges'].append(edge)

        node_info = get_node_information(global_topology, node)
        if node_info not in sub_topology['nodes']:
            sub_topology['nodes'].append(node_info)
        for nb in neighbors:
            nb_info = get_node_information(global_topology, nb)
            if nb_info not in sub_topology['nodes']:
                sub_topology['nodes'].append(nb_info)
    return sub_topology

def get_routemap_config(configs):
    routemap_config = ''
    config = configs
    route_map_config = []
    segments = config.split('!')
    for seg in segments:
        if seg == '':
            continue
        if 'router bgp' not in seg and 'interface' not in seg:
            route_map_config.append(seg)
    routemap_config += '!'.join(route_map_config)

    return routemap_config

def get_bgp_routemap_config(configs):
    routemap_config = ''
    bgpconfig = ''
    bgp_config = []
    route_map_config = []
    segments = configs.split('!')
    for seg in segments:
        if seg == '' or seg == '\n':
            continue
        if 'router bgp' in seg:
            bgp_config.append(seg)
        elif 'interface' in seg:
            continue
        else:
            route_map_config.append(seg)
    routemap_config += '!'.join(route_map_config)
    bgpconfig += '!'.join(bgp_config)
    config = routemap_config + '!\n!' + bgpconfig

    return config


def clean_config(config_text):
    return re.sub(r'(!\n)+', '!\n', config_text)


def generate_configuration_basedLLM_OSPF(nodes, sub_target_topo, user_intent, configurationExample, associations, intent_id, folder_path):
    with open('./prompts/configuration_prefix.txt', 'r', encoding='utf-8') as file:
        prefix_prompt = file.read()

    associated_nodes = []
    for user_node, exp_node, in associations.items():
        if exp_node not in associated_nodes:
            associated_nodes.append(exp_node)

    with open('./prompts/generation/example.txt', 'r', encoding='utf-8') as file:
        example_prompt_template = file.read()

    example_topology = configurationExample['topology']
    example_intent = configurationExample['intent']

    config_prompt = ''
    for match_node in associated_nodes:
        full_config = configurationExample['configurations'][match_node]
        full_config = clean_config(full_config)
        config = full_config
        config_prompt = config_prompt + 'Configuration of ' + match_node + ': \n' + config + '\n\n'

    example_prompt = example_prompt_template.format(example_topology=example_topology, example_intent=example_intent, example_configurations=config_prompt)

    instruct_path = f'./prompts/instructions/ospf_complete_instruct.txt'
    with open(instruct_path, 'r', encoding='utf-8') as file:
        instruction = file.read()

    with open('./prompts/generation/targetScenario.txt', 'r') as f:
        target_prompt_template = f.read()

    target_prompt = target_prompt_template.format(target_topology=sub_target_topo, user_intent=user_intent, associations=associations)

    with open('./prompts/generation/suffix.txt', 'r') as f:
        suffix_template = f.read()

    suffix_prompt = suffix_template.format(nodes=nodes, instruction=instruction)

    prompt = prefix_prompt + '\n\n' + example_prompt + '\n\n' + target_prompt + '\n\n' + suffix_prompt

    #print('configuration prompt', prompt)
    answer = ask_LLM(prompt)
    #print('answer', answer)
    file_path = folder_path + '/config' + str(intent_id) + '.txt'
    with open(file_path, 'a', encoding='utf-8') as file:
        file.write(answer)
    with open(file_path, 'a', encoding='utf-8') as file:
        file.write('\n\n')

    return prefix_prompt

def generate_configuration_basedLLM_BGP(nodes, sub_target_topo, user_intent, configurationExample, associations, intent_id, folder_path):
    with open('./prompts/configuration_prefix.txt', 'r', encoding='utf-8') as file:
        prefix_prompt = file.read()

    associated_nodes = []
    for user_node, exp_node, in associations.items():
        if exp_node not in associated_nodes:
            associated_nodes.append(exp_node)

    with open('./prompts/generation/example.txt', 'r', encoding='utf-8') as file:
        example_prompt_template = file.read()

    example_topology = configurationExample['topology']
    example_intent = configurationExample['intent']

    config_prompt = ''

    for match_node in associated_nodes:
        full_config = configurationExample['configurations'][match_node]
        full_config = clean_config(full_config)
        config = full_config
        config_prompt = config_prompt + 'Configuration of ' + match_node + ': \n' + config + '\n\n'
    instruct_path = f'./prompts/instructions/bgp_complete_instruction.txt'
    with open(instruct_path, 'r', encoding='utf-8') as file:
        instruction = file.read()

    example_prompt = example_prompt_template.format(example_topology=example_topology, example_intent=example_intent,
                                                    example_configurations=config_prompt)

    with open('./prompts/generation/targetScenario.txt', 'r') as f:
        target_prompt_template = f.read()

    target_prompt = target_prompt_template.format(target_topology=sub_target_topo, user_intent=user_intent,
                                                  associations=associations)

    with open('./prompts/generation/suffix.txt', 'r') as f:
        suffix_template = f.read()

    suffix_prompt = suffix_template.format(nodes=nodes, instruction=instruction)

    prompt = prefix_prompt + '\n\n' + example_prompt + '\n\n' + target_prompt + '\n\n' + suffix_prompt

    #print('configuration prompt', prompt)
    answer = ask_LLM(prompt)
    #print('answer', answer)
    file_path = folder_path + '/config' + str(intent_id) + '.txt'
    with open(file_path, 'a', encoding='utf-8') as file:
        file.write(answer)
    with open(file_path, 'a', encoding='utf-8') as file:
        file.write('\n\n')

    return prefix_prompt

def convert_configuration_to_json(data):
    # Updated regular expression to capture all text until the next "Configuration template" or end of string
    pattern = r"Configuration of ([^:]+):([\s\S]+?)(?=Configuration of |\Z)"

    matches = re.findall(pattern, data)
    configurations = {name.strip(): content.strip() for name, content in matches}

    return json.dumps(configurations, indent=4)


def synthesizeConfiguration(intent_id, user_intent, target_topo, protocol, configExample, assocationRelations, batch_size, folder_path):
    nodes = []
    for node_dict in target_topo['nodes']:
        #print('node_dict', node_dict)
        nodes.append(node_dict['name'])
    node_num = len(nodes)
    parts = math.ceil(node_num / batch_size)
    #print('parts', parts)

    sub_nodes_set = []
    subtopo_set = []
    subassociations_set = []
    #print('associations', assocationRelations)
    for i in range(parts):
        sub_nodes = nodes[i * batch_size:min((i + 1) * batch_size, len(nodes))]
        sub_nodes_set.append(sub_nodes)

        sub_associations = {}

        for key in sub_nodes:
            sub_associations[key] = assocationRelations[key]

        subassociations_set.append(sub_associations)

        subtopo = get_sub_topology(target_topo, sub_nodes)
        subtopo_set.append(subtopo)

    future_tasks = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        for i in range(len(subtopo_set)):
            # print('args', args)
            args = [sub_nodes_set[i], subtopo_set[i], user_intent, configExample, subassociations_set[i], intent_id, folder_path]
            if protocol == 'OSPF':
                future_task = executor.submit(generate_configuration_basedLLM_OSPF, *args)
            elif protocol == 'BGP':
                future_task = executor.submit(generate_configuration_basedLLM_BGP, *args)
            elif protocol == 'Static':
                continue
                #future_task = executor.submit(generate_configuration_basedLLM_Static, *args)
            else:
                raise ValueError("Unsupported routing protocol %s", protocol)
            future_tasks.append(future_task)

        prompt = future_tasks[0].result()

    while True:
        file_path = folder_path + '/config' + str(intent_id) + '.txt'
        with open(file_path, 'r', encoding='utf-8') as file:
            configs = file.read()

        config_json = convert_configuration_to_json(configs)
        if type(config_json) == str:
            config_json = eval(config_json)
        node_with_configs = list(config_json.keys())

        unconfigured_nodes = list(set(nodes) - set(node_with_configs))
        if len(unconfigured_nodes) > 0:
            #print('!!!unconfigured_nodes', unconfigured_nodes)
            sub_associatios = {}
            associated_nodes = []
            for key in unconfigured_nodes:
                sub_associatios[key] = assocationRelations[key]

            subtopo = get_sub_topology(target_topo, unconfigured_nodes)
            if protocol == 'BGP':
                generate_configuration_basedLLM_BGP(unconfigured_nodes, subtopo, user_intent, configExample, sub_associatios, intent_id, folder_path)
            elif protocol == 'OSPF':
                generate_configuration_basedLLM_OSPF(unconfigured_nodes, subtopo, user_intent, configExample,
                                                    sub_associatios, intent_id, folder_path)
        else:
            break

    return config_json



def generate_edge_cost(intents):
    formal_intents = []
    intent_types = []
    for intent in intents:
        user_specification_intent = intent_to_formalspecifications(intent)
        user_specification_list = extract_specification(user_specification_intent)
        formal_intents.append(user_specification_list[0])
        intent_type = parse_intentType(user_specification_list[0])
        intent_types.append(intent_type)
    intent_types = list(set(intent_types))

    #print('!!!formal_intents', formal_intents)
    if len(formal_intents) == 0:
        raise ValueError("no formal intents")
    if len(intent_types) > 1:
        raise ValueError("Unsupported intent types")

    req_type = intent_types[0]

    edge_costs = ospfeval(req_type, formal_intents)
    #print('!!!edge_costs', edge_costs)

    return edge_costs


def parse_interface_config(config_text):
    interface_pattern = re.compile(r'(interface \S+)(.*?)(?=^\s*interface|\Z)', re.S | re.M)
    interface_dict = {}

    matches = interface_pattern.findall(config_text)
    for match in matches:
        interface_name = match[0].strip()
        interface_config = match[1].strip()

        interface_config_full = f'{interface_name}\n{interface_config}'
        interface_dict[interface_name] = interface_config_full

    return interface_dict


def get_interface_cost(topo, node, interface_name, interface_cost):
    interface_name = interface_name.split(' ')[1]
    #print('get_interface_cost', interface_cost)
    #print('node', node, interface_name)
    for link in topo['edges']:
        node1 = link['node1']
        node2 = link['node2']
        if node1['name'] == node and node1['interface'] == interface_name:
            neighbor = node2['name']
            #print('neighbor', neighbor, interface_cost[neighbor])
            if neighbor in interface_cost:
                return interface_cost[neighbor]
            else:
                return 0
        elif node2['name'] == node and node2['interface'] == interface_name:
            neighbor = node1['name']
            if neighbor in interface_cost:
                return interface_cost[neighbor]
            else:
                return 0


def fill_ospf_cost(topo, interface_configs, edge_costs):
    changed_interface_configs = {}
    for node, interface_config in interface_configs.items():

        interface_config_dict = parse_interface_config(interface_config)

        if node in edge_costs:
            interface_cost = edge_costs[node]
            changed_interface_configs_node = []
            for interface_name, config in interface_config_dict.items():
                cost = get_interface_cost(topo, node, interface_name, interface_cost)
                if not cost:
                    cost = 0
                pattern = r'ip ospf cost \d+'
                updated_config = re.sub(pattern, f'ip ospf cost {cost}', config)
                changed_interface_configs_node.append(updated_config)
            changed_interface_configs[node] = '\n'.join(changed_interface_configs_node)
        else:
            changed_interface_configs[node] = interface_config

    return changed_interface_configs
