import json
import re
import os
import shutil
import torch
from querier import *
from classifier import *
from generator import *
from Semantic_verifier import *
from sentence_transformers import SentenceTransformer
from utils import GraphSAGE
from querier import FastTextFeatureExtractor
from setting_manager import setting

def parseIntents(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    pattern = r'\d+\.\s*(.*?)(?=\d+\.|$)'
    sentences = re.findall(pattern, content, re.DOTALL)
    sentences = [sentence.strip() for sentence in sentences]
    return sentences

def main():
    model_paths = setting.get_model_paths()
    output_paths = setting.get_output_paths()
    input_paths = setting.get_input_paths()
    training_config = setting.get_training_config()
    
    gcn_model = GraphSAGE(in_channels=300, hidden_size=256, out_channels=128, num_layers=2)
    checkpoint = torch.load(model_paths['gcn'], map_location=torch.device('cpu'))
    gcn_model.load_state_dict(checkpoint['gcn_state_dict'])
    batch_size = training_config['batch_size']

    folder_path = output_paths['responses']
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.mkdir(folder_path)
    os.makedirs(f'{folder_path}/configurations')
    os.makedirs(f'{folder_path}/route_map_results')

    output_file = output_paths['output']
    if os.path.exists(output_file):
        shutil.rmtree(output_file)
    os.mkdir(output_file)
    os.makedirs(f'{output_file}/configs')

    LAV = local_attribute_verifier(folder_path)
    GFV = global_formal_verifier(folder_path)
    loops = 0

    role_text_model = FastTextFeatureExtractor()

    user_intents = parseIntents(input_paths['intent'])

    with open(input_paths['topology'], 'r') as f:
        target_topo = json.load(f)

    # sort intents according to routing protocols: OSPF, BGP
    classified_intents = {'OSPF':[], 'BGP': [], 'Static': []}
    intent_id = 1
    for intent in user_intents:
        normalized_user_intent = process_intent(intent)
        protocol = normalized_user_intent.split(' ')[0]
        intent_dict = {}
        intent_dict['original intent'] = intent
        intent_dict['normalized intent'] = normalized_user_intent
        intent_dict['id'] = intent_id
        intent_id += 1
        classified_intents[protocol].append(intent_dict)

    #print('classified_intents', classified_intents)

    if len(classified_intents['OSPF']) > 0:
        num = 0
        for intent_dict in classified_intents['OSPF']:
            intent_id = intent_dict['id']
            num += 1
            if num > 1:
                break
            original_intent = intent_dict['original intent']
            print(f'***Generate configuration for intent {intent_id}: {original_intent}')
            normalized_intent = intent_dict['normalized intent']
            configExample, user_intentGraph, example_intentGraph = retrievalExamples(original_intent, normalized_intent, target_topo)
            associationRelations = associateNodes(user_intentGraph, example_intentGraph, gcn_model, role_text_model)
            node_configurations = synthesizeConfiguration(intent_id, original_intent, target_topo, 'OSPF',
                                                          configExample, associationRelations, batch_size,
                                                          folder_path)
            loops += verify_configurations('OSPF', LAV, GFV, node_configurations, intent_id, original_intent, target_topo,
                                  associationRelations, configExample)

    if len(classified_intents['BGP']) > 0:
        all_bgp_intents = []
        num = 0
        configExample = None
        all_examples = []
        all_associations = []
        for intent_dict in classified_intents['BGP']:
            intent_id = intent_dict['id']
            num += 1
            original_intent = intent_dict['original intent']
            print(f'***Generate configuration for intent {intent_id}: {original_intent}')
            all_bgp_intents.append(original_intent)
            normalized_intent = intent_dict['normalized intent']
            configExample, user_intentGraph, example_intentGraph = retrievalExamples(original_intent, normalized_intent, target_topo)
            all_examples.append(configExample)
            associationRelations = associateNodes(user_intentGraph, example_intentGraph, gcn_model, role_text_model)
            all_associations.append(associationRelations)
            node_configurations = synthesizeConfiguration(intent_id, original_intent, target_topo, 'BGP',
                                                          configExample, associationRelations, batch_size,
                                                          folder_path)
            loops += verify_configurations('BGP', LAV, GFV, node_configurations, intent_id, original_intent, target_topo,
                                  associationRelations, configExample)


        # merge configs of all BGP intents
        merged_configs = merge_templates(f'{folder_path}/route_map_results')
        file_path = folder_path + '/merged.json'
        with open(file_path, 'w') as file:
            json.dump(merged_configs, file)

        all_routing_policy_configs = json_to_config(merged_configs)
        file_path = folder_path + '/merged.txt'
        with open(file_path, 'w') as file:
            file.write(all_routing_policy_configs)

        loops += GFV.verify_bgp_routing_policy(10001, all_bgp_intents, target_topo, merged_configs, all_associations,
                                              all_examples)

        print('total loops', loops)

    # write configurations
    interface_configs = {}
    ospf_configs = {}
    iproute_configs = {}
    bgp_configs = {}
    route_map_configs = {}

    interface_config_path = f'{folder_path}/configurations/interface.json'
    ospf_config_path = f'{folder_path}/configurations/ospf.json'
    iproute_config_path = f'{folder_path}/configurations/iproute.json'
    bgp_config_path = f'{folder_path}/configurations/bgp.json'
    route_map_config_path = f'{folder_path}/configurations/route_map_configs.json'
    
    if os.path.exists(interface_config_path):
        with open(interface_config_path, 'r') as file:
            interface_configs = json.load(file)

    if os.path.exists(ospf_config_path):
        with open(ospf_config_path, 'r') as file:
            ospf_configs = json.load(file)

    if os.path.exists(iproute_config_path):
        with open(iproute_config_path, 'r') as file:
            iproute_configs = json.load(file)

    if os.path.exists(bgp_config_path):
        with open(bgp_config_path, 'r') as file:
            bgp_configs = json.load(file)

    if os.path.exists(route_map_config_path):
        with open(route_map_config_path, 'r') as file:
            route_map_configs = json.load(file)

    #generate link cost
    if len(classified_intents['OSPF']) > 0:
        edge_costs = generate_edge_cost(classified_intents['OSPF'])
        if edge_costs:
            interface_configs = fill_ospf_cost(target_topo, interface_configs, edge_costs)

    configs_folder = output_paths['configs']
    for node, interface_config in interface_configs.items():
        ospf_config = ''
        bgp_config = ''
        route_map_config = ''
        if node in ospf_configs and 'Peer' not in node:
            ospf_config = ospf_configs[node]
        if node in bgp_configs:
            bgp_config = bgp_configs[node]
        if node in route_map_configs:
            route_map_config = route_map_configs[node]

        combined_config = route_map_config + bgp_config + '\n!'
        route_map_config_json, bgp_config_json = parse_router_config(combined_config)
        new_bgp_config_json = update_bgp_config(target_topo, route_map_config_json, bgp_config_json)
        new_bgp_config = convert_json_to_bgp(new_bgp_config_json)
        if node in list(iproute_configs.keys()):
            iproute_config = iproute_configs[node]
        else:
            iproute_config = ''

        if iproute_config != '':
            node_config = f'hostname {node}\n!\n' + interface_config + '\n!\n!\n' + ospf_config + '\n!\n!\n' + iproute_config + '\n!\n!\n' + route_map_config + '\n!\n!\n' + new_bgp_config + '\n!'
        else:
            node_config = f'hostname {node}\n!\n' + interface_config + '\n!\n!\n' + ospf_config + '\n!\n!\n' + route_map_config + '\n!\n!\n' + new_bgp_config + '\n!'
        cfg_file = configs_folder + f'{node}.cfg'
        with open(cfg_file, 'w') as fhandle:
            fhandle.write(node_config)


if __name__ == "__main__":
    main()