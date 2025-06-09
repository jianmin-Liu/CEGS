import ast
import collections
import copy
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import torch
from torch_geometric.data import Data
import torch.nn as nn
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from netcomplete.netcomplete_ebgp_eval import *
import google.generativeai as genai
from setting_manager import setting
import os

llm_provider = setting.get_llm_provider()

if llm_provider == "OPENAI":
    from openai import OpenAI
    openai_config = setting.get_openai_config()
    
    client = OpenAI(
        api_key=openai_config['api_key']
    )
    
    def ask_LLM(prompt: str):
        try:
            system_intel = "You are a helpful assistant."
            response = client.chat.completions.create(
                model=openai_config['model'],
                messages=[
                    {"role": "system", "content": system_intel},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return None

elif llm_provider == "GEMINI":
    import google.generativeai as genai
    
    gemini_config = setting.get_gemini_config()
    os.environ['OPENAI_API_KEY'] = gemini_config['api_key']
    genai.configure(api_key=os.environ['OPENAI_API_KEY'])
    
    def ask_LLM(prompt: str):
        try:
            model = genai.GenerativeModel(gemini_config['model'])
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Gemini API error: {e}")
            return None

elif llm_provider == "DEEPSEEK":
    from openai import OpenAI
    deepseek_config = setting.get_deepseek_config()
    
    client = OpenAI(
        api_key=deepseek_config['api_key'],
        base_url="https://api.deepseek.com/v1"
    )
    
    def ask_LLM(prompt: str):
        try:
            system_intel = "You are a helpful assistant."
            response = client.chat.completions.create(
                model=deepseek_config['model'],
                messages=[
                    {"role": "system", "content": system_intel},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"DeepSeek API error: {e}")
            return None

else:
    raise ValueError(f"Unsupported LLM provider: {llm_provider}")


def _canonical_representation(obj, visited=None):
    """get the canonical representation for voting comparison, avoid circular references"""
    if visited is None:
        visited = set()
    
    # check object ID to avoid circular references
    obj_id = id(obj)
    if obj_id in visited:
        return f"<Circular reference: {type(obj).__name__}>"
    
    if isinstance(obj, dict):
        visited.add(obj_id)
        try:
            result = tuple(sorted((k, _canonical_representation(v, visited)) for k, v in obj.items()))
        finally:
            visited.remove(obj_id)
        return result
    elif isinstance(obj, list):
        visited.add(obj_id)
        try:
            result = tuple(_canonical_representation(x, visited) for x in obj)
        finally:
            visited.remove(obj_id)
        return result
    elif hasattr(obj, '__class__') and hasattr(obj, '__dict__'):
        # handle custom class objects 
        visited.add(obj_id)
        try:
            class_name = obj.__class__.__name__
            # convert object attributes to hashable representation, but limit recursion depth
            attrs = tuple(sorted((k, _canonical_representation(v, visited)) for k, v in obj.__dict__.items()))
            result = (class_name, attrs)
        finally:
            visited.remove(obj_id)
        return result
    elif hasattr(obj, '__class__'):
        # if object has no __dict__ attribute, use string representation
        return (obj.__class__.__name__, str(obj))
    else:
        return obj


class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_size, out_channels, num_layers):
        super(GraphSAGE, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_size))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_size, hidden_size))
        self.convs.append(SAGEConv(hidden_size, out_channels))

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


def get_connectToPath_nodes(graph, path):
    nodes = []
    for link in graph['links']:
        node1 = link['node1']['name']
        node2 = link['node2']['name']
        if node1 in path and node2 not in path:
            nodes.append(node2)
        elif node1 not in path and node2 in path:
            nodes.append(node1)
    return nodes


def llm_output_to_json(llm_output_str):
    """extract JSON data from string, support Markdown code block format"""
    try:
        return json.loads(llm_output_str)
    except json.JSONDecodeError:
        pass
    pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    matches = re.findall(pattern, llm_output_str)

    if matches:
        json_str = matches[0]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON from code block: {e}")

    raise ValueError("No valid JSON found in the input string")


def get_node_attrs_oreder(intents, topo):
    path1 = intents[0]
    path2 = intents[1]
    roles = {}

    for node in topo['routers']:
        if node == path1[0]:
            role = 'route preference'
        elif node == path1[-1]:
            role = 'destination'
        elif node in path1[1:-1] or node in path2[1:-1]:
            role = 'relay'
        else:
            role = 'non-involvement'
        roles[node] = role
    return roles


def get_node_attrs_ECMP(intents, topo):
    path1 = intents[0]
    path2 = intents[1]
    roles = {}

    for node in topo['routers']:
        if node == path1[0]:
            role = 'load balance'
        elif node == path1[-1]:
            role = 'destination'
        elif node in path1[1:-1] or node in path2[1:-1]:
            role = 'relay'
        else:
            role = 'non-involvement'
        roles[node] = role
    return roles


def get_node_attrs_Anypath(intents, topo):
    path1 = intents[0]
    path2 = intents[1]
    roles = {}

    for node in topo['routers']:
        if node == path1[0]:
            role = 'source'
        elif node == path1[-1]:
            role = 'destination'
        elif node in path1[1:-1] or node in path2[1:-1]:
            role = 'relay'
        else:
            role = 'non-involvement'
        roles[node] = role
    return roles


def get_node_attrs_simple(intents, topo):
    path = intents
    roles = {}

    for node in topo['routers']:
        if node == path[0]:
            role = 'source'
        elif node == path[-1]:
            role = 'destination'
        elif node in path[1:-1]:
            role = 'relay'
        else:
            role = 'non-involvement'
        roles[node] = role
    return roles

def get_roles(type, intent, topology):
    if type == 'order':
        return get_node_attrs_oreder(intent, topology)
    elif type == 'ecmp':
         return get_node_attrs_ECMP(intent, topology)
    elif type == 'kconnected':
        get_node_attrs_Anypath(intent, topology)
    elif type == 'simple':
        get_node_attrs_simple(intent, topology)


# def generateGraph(intent_type, intent, topology):
#     """generate intent graph"""
#     num_samples = setting.get("GENERATE_GRAPH_SAMPLES", 5)
#     max_workers = setting.get("SELF_CONSISTENCY_PARALLEL_WORKERS", 5)
#
#     role_prompt_dir = setting.get_intent_types_role_prompt_dir()
#     with open(f'{role_prompt_dir}/{intent_type}.txt', 'r') as f:
#         prefix = f.read()
#
#     prompt = prefix + '\n\n' + f'Intent:{intent}\nTopology:{topology}'
#
#     responses = []
#     #print(f"Using {max_workers} parallel workers to generate {num_samples} node role samples")
#
#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         futures = []
#         for i in range(num_samples):
#             futures.append(executor.submit(ask_LLM, prompt))
#
#         for idx, future in enumerate(as_completed(futures)):
#             try:
#                 roles_str = future.result()
#                 if roles_str:
#                     roles_json = llm_output_to_json(roles_str)
#                     if roles_json:
#                         responses.append(roles_json)
#                     else:
#                         print(f"sample {idx + 1} parsing failed")
#                 else:
#                     print(f"sample {idx + 1} LLM returned empty")
#             except Exception as e:
#                 print(f"sample execution failed: {str(e)}")
#
#     if not responses:
#         print("fail to generate intent graph")
#         return None
#
#     # Self-Consistency: vote for the best answer
#     vote_counts = collections.Counter()
#     for response in responses:
#         canonical_repr = _canonical_representation(response)
#         vote_counts[canonical_repr] += 1
#
#     if not vote_counts:
#         return None
#
#     # Select the answer with the most votes
#     most_common_repr, votes = vote_counts.most_common(1)[0]
#     #print(f"Self-Consistency voting result: highest votes {votes}/{len(responses)} from {num_samples} samples")
#
#     chosen_roles = None
#     for response in responses:
#         if _canonical_representation(response) == most_common_repr:
#             chosen_roles = response
#             break
#
#     if not chosen_roles:
#         chosen_roles = responses[0]
#
#     print('roles', chosen_roles)
#
#     # build intent graph
#     graph = {}
#     graph['nodes'] = chosen_roles
#     graph['edges'] = []
#     for edge in topology['edges']:
#         node1 = edge['node1']['name']
#         node2 = edge['node2']['name']
#         graph['edges'].append(f'{node1}_{node2}')
#
#     return graph


def generateGraph(intent_type, intent, topology):
    role_prompt_dir = setting.get_intent_types_role_prompt_dir()
    with open(f'{role_prompt_dir}/{intent_type}.txt', 'r', encoding="utf-8") as f:
        prefix = f.read()

    prompt = prefix + '\n\n' + f'Intent:{intent}\nTopology:{topology}'

    roles_str = ask_LLM(prompt)
    roles_json = llm_output_to_json(roles_str)
    graph = {}
    graph['nodes'] = roles_json
    graph['edges'] = []
    for edge in topology['edges']:
        node1 = edge['node1']['name']
        node2 = edge['node2']['name']
        graph['edges'].append(f'{node1}_{node2}')

    return graph


def create_graph_data(graph, text_model):
    node_features = []
    node_names = list(graph['nodes'].keys())
    if not node_names:
        return None

    node_to_idx = {name: i for i, name in enumerate(node_names)}

    for node in graph['nodes'].keys():
        role = graph['nodes'][node]
        #role_embedding = text_model.encode_text(role)
        role_embedding = text_model.encode(role)
        node_features.append(torch.tensor(role_embedding, dtype=torch.float32))

    # create edge indices
    edge_list = []
    for edge_key in graph.get('edges', {}):
        edge_parts = edge_key.split('_', 1)
        if len(edge_parts) == 2:
            source, target = edge_parts
            if source in node_to_idx and target in node_to_idx:
                edge_list.append([node_to_idx[source], node_to_idx[target]])

    x = torch.stack(node_features) if node_features else torch.empty((0, len(node_features[0])))
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous() if edge_list else torch.empty((2, 0),
                                                                                                          dtype=torch.long)
    data = Data(x=x, edge_index=edge_index)

    return data



def get_neighbors(topo, node, paths):
    #all_paths = paths[0] + paths[1]
    neighbors = {}
    for link in topo['links']:
        nb = None
        if link['node1']['name'] == node:
            nb = link['node2']['name']
        elif link['node2']['name'] == node:
            nb = link['node1']['name']
        if nb:
            if nb in paths[0] and node in paths[0]:
                nb_id = paths[0].index(nb)
                node_id = paths[0].index(node)
                if nb_id < node_id:
                    neighbors[nb] = 'last node'
                else:
                    neighbors[nb] = 'next node'
            elif nb in paths[1] and node in paths[1]:
                nb_id = paths[1].index(nb)
                node_id = paths[1].index(node)
                if nb_id < node_id:
                    neighbors[nb] = 'last node'
                else:
                    neighbors[nb] = 'next node'
            else:
                neighbors[nb] = 'connected'
    return neighbors


def process_community(intent_id, configs):
    segments = configs.split('!\n')
    tmp_segs = []
    for seg in segments:
        lines = seg.split('\n')
        tmp_lines = []
        for line in lines:
            if line == ' ':
                continue
            if 'match community' in line:
                tmp_lines.append(f'match community {intent_id}')
            else:
                tmp_lines.append(line)
        tmp_segs.append('\n'.join(tmp_lines))
    return '!'.join(tmp_segs)

def process_templates(intent_id, configs):
    segments = configs.split('!')
    segments_ = copy.deepcopy(segments)
    pattern1 = r'route-map (RMap_[^\s]+)_from_([^\s]+) permit'
    pattern2 = r'route-map (RMap_[^\s]+)_from_([^\s]+) deny'

    for i in range(len(segments) - 1):
        segment = segments[i]
        segment_next = segments[i + 1]
        matches1 = re.findall(pattern1, segment)
        matches2 = re.findall(pattern2, segment_next)
        if matches1 == matches2 and matches1:
            segments_.remove(segment_next)
            temp = segment + segment_next[1:]
            segments_[i] = temp
    return process_community(intent_id, '!\n'.join(segments_))

def parse_route_map(route_map_text):
    """Parse route-map text into a list of entries in order"""
    entries = []
    current_entry = None

    for line in route_map_text.split('\n'):
        line = line.strip()
        if not line:
            continue

        # Check for route-map header line
        match = re.match(r'route-map\s+(\S+)\s+(permit|deny)\s+(\d+)', line)
        if match:
            if current_entry is not None:
                entries.append(current_entry)

            current_entry = {
                'name': match.group(1),
                'action': match.group(2),
                'seq': int(match.group(3)),
                'lines': [line],
                'has_match': False
            }
        elif current_entry is not None:
            current_entry['lines'].append(line)
            if line.startswith('match'):
                current_entry['has_match'] = True

    # Add the last entry
    if current_entry is not None:
        entries.append(current_entry)

    return entries[0]['name'] if entries else None, entries


def starts_with_route_map(text):
    # Check if the first line matches the pattern
    first_line = text.split('\n')[0].strip()
    pattern = r'^route-map\s+(\S+)\s+(permit|deny)\s+(\d+)$'
    return bool(re.fullmatch(pattern, first_line))

def extract_routemap_configs(configs):
    route_map_configs = {}
    for node, config in configs.items():
        segments = config.split('!\n')
        #print('segments', segments)
        route_map_config = []
        for seg in segments:
            if starts_with_route_map(seg):
                route_map_config.append(seg)
        routemap_config = '!\n'.join(route_map_config)
        route_map_configs[node] = routemap_config

    return route_map_configs


def merge_same_map(template1, template2):
    """Merge two route-map templates"""
    name1, entries1 = parse_route_map(template1)
    name2, entries2 = parse_route_map(template2)

    # Create a combined list of all unique entries
    combined = []
    seen_entries = set()

    # Helper function to get entry content signature
    def get_signature(entry):
        # Ignore the sequence number when comparing content
        return '\n'.join(entry['lines'][1:])

    # Add entries from first template
    for entry in entries1:
        sig = get_signature(entry)
        if sig not in seen_entries:
            combined.append(entry)
            seen_entries.add(sig)

    # Add entries from second template
    for entry in entries2:
        sig = get_signature(entry)
        if sig not in seen_entries:
            combined.append(entry)
            seen_entries.add(sig)

    # Separate entries with match and without match
    with_match = [e for e in combined if e['has_match']]
    without_match = [e for e in combined if not e['has_match']]

    # Combine them with match entries first
    sorted_combined = with_match + without_match

    # Renumber entries sequentially starting from 10 with step 10
    for i, entry in enumerate(sorted_combined, start=1):
        new_seq = i * 10
        # Update the sequence number in the header line
        header = entry['lines'][0]
        new_header = re.sub(r'(route-map\s+\S+\s+(permit|deny)\s+)\d+',
                            f'\\g<1>{new_seq}', header)
        entry['lines'][0] = new_header
        entry['seq'] = new_seq

    # Generate the merged output
    output = []
    for entry in sorted_combined:
        output.extend(entry['lines'])

    return '\n'.join(output)


def merge(template1, template2):
    segments1 = template1.split('!')
    maps1 = {}
    for seg in segments1:
        if seg == '\n' or seg == '':
            continue
        if not seg.startswith('\n'):
            seg = '\n' + seg
        if not seg.endswith('\n'):
            seg = seg + '\n'
        lines_ = seg.split('\n')
        lines = [element for element in lines_ if element != '']
        if len(lines) > 0:
            if len(re.findall(r'route-map\s+(\S+)\s+([\s\S]+?)', lines[0])) > 0:
                route_map_name = re.findall(r'route-map\s+(\S+)\s+([\s\S]+?)', lines[0])[0][0]
                maps1[route_map_name] = seg

    segments2 = template2.split('!')
    if segments2 == ['\n\n']:
        return template1
    maps2 = {}
    for seg in segments2:
        if seg == '\n' or seg == '':
            continue
        if not seg.startswith('\n'):
            seg = '\n' + seg
        if not seg.endswith('\n'):
            seg = seg + '\n'
        lines_ = seg.split('\n')
        lines = [element for element in lines_ if element != '']
        if len(re.findall(r'route-map\s+(\S+)\s+([\s\S]+?)', lines[0])) > 0:
            route_map_name = re.findall(r'route-map\s+(\S+)\s+([\s\S]+?)', lines[0])[0][0]
            maps2[route_map_name] = seg

    merged_maps = copy.deepcopy(maps1)
    for map in maps2:
        if map not in maps1:
            merged_maps[map] = maps2[map]
        else:
            if maps2[map] != maps1[map]:
                pattern1 = maps1[map].split('\n')
                pattern1 = list(filter(None, pattern1))
                pattern2 = maps2[map].split('\n')
                pattern2 = list(filter(None, pattern2))
                if len(pattern1) == 1 and len(pattern2) > 1:
                    merged_maps[map] = maps2[map]
                elif len(pattern1) > 1 and len(pattern2) > 1:
                    merged_maps[map] = merge_same_map(maps1[map], maps2[map])
            else:
                pattern1 = maps1[map].split('\n')
                pattern1 = list(filter(None, pattern1))
                pattern2 = maps2[map].split('\n')
                pattern2 = list(filter(None, pattern2))
                if len(pattern1) == 1 and len(pattern2) > 1:
                    merged_maps[map] = maps2[map]
                elif len(pattern1) > 1 and len(pattern2) > 1:
                    merged_maps[map] = merge_same_map(maps1[map], maps2[map])
    return '!'.join(['\n', *merged_maps.values(), '\n'])


def merge_node_templates(node, templates):
    template = templates['intent1'][node]
    template = process_templates(1, template)
    for i in range(1, len(templates.keys())):
        if node not in templates['intent' + str(i + 1)].keys():
            continue
        if templates['intent' + str(i + 1)][node] == '':
            continue
        intent_id = i + 1
        node_template = process_templates(intent_id, templates['intent'+str(i+1)][node])
        template = merge(template, node_template)
    return template

def merge_templates(folder_path):
    templates = {}
    id = 1
    for file in os.listdir(folder_path):
        if 'result' not in file:
            continue
        file_path = folder_path + '/' + file
        with open(file_path, 'r', encoding='utf-8') as file:
            template_json = json.load(file)
            template_json = extract_routemap_configs(template_json)
        templates['intent'+str(id)] = template_json
        id += 1

    merged_templates = {}
    for node in list(templates['intent1'].keys()):
        merged_templates[node] = '!\n' + merge_node_templates(node, templates) + '\n!'
    return merged_templates


def parse_router_config(config):
    route_map_configs = {}
    bgp_configs = {"bgp": ""}
    current_route_map = None
    current_bgp_neighbor = None
    in_route_map = False
    in_bgp = False

    # Split the config into lines
    lines = config.splitlines()

    # Regex patterns
    route_map_pattern = re.compile(r'^route-map (\S+)(.*)')
    bgp_pattern = re.compile(r'^router bgp (\d+)')
    neighbor_pattern = re.compile(r'^neighbor (\S+)')

    # Parse the input config
    for line in lines:
        stripped_line = line.strip()

        # Check for route-map start
        route_map_match = route_map_pattern.match(stripped_line)
        if route_map_match:
            current_route_map = route_map_match.group(1)
            route_map_configs.setdefault(current_route_map, "")
            route_map_configs[current_route_map] += line + "\n"
            in_route_map = True
            in_bgp = False
            continue

        # Check for router bgp start
        bgp_match = bgp_pattern.match(stripped_line)
        if bgp_match:
            bgp_configs["bgp"] += line + "\n"
            in_bgp = True
            in_route_map = False
            current_bgp_neighbor = None
            continue

        # Check for neighbor statement in bgp config
        neighbor_match = neighbor_pattern.match(stripped_line)
        if neighbor_match and in_bgp:
            current_bgp_neighbor = neighbor_match.group(1)
            bgp_configs.setdefault(current_bgp_neighbor, "")
            bgp_configs[current_bgp_neighbor] += line + "\n"
            continue

        # Add lines to the current context
        if in_route_map and current_route_map:
            route_map_configs[current_route_map] += line + "\n"
        elif in_bgp:
            if current_bgp_neighbor:
                bgp_configs[current_bgp_neighbor] += line + "\n"
            else:
                bgp_configs["bgp"] += line + "\n"

    #Remove any trailing newlines
    for key in route_map_configs:
        route_map_configs[key] = route_map_configs[key].strip()

    for key in bgp_configs:
        #bgp_configs[key] = bgp_configs[key].strip()
        if key == 'bgp':
            bgp_configs[key] = bgp_configs[key].strip()
        else:
            bgp_configs[key] = ' ' + bgp_configs[key].strip()

    return route_map_configs, bgp_configs

def get_neighbor_ip(topo, node, neighbor):
    for link in topo['edges']:
        node1 = link['node1']['name']
        node2 = link['node2']['name']
        if node == node1 and neighbor == node2:
            return link['node2']['ip address'].split('/')[0]
        if node == node2 and neighbor == node1:
            return link['node1']['ip address'].split('/')[0]

def update_bgp_config(topo, route_maps, bgp_neighbors):
    route_map_reference_pattern = re.compile(r'route-map (\S+) (in|out)')
    for neighbor, neighbor_config in bgp_neighbors.items():
        new_neighbor_config = []
        commands = neighbor_config.split('\n')
        for cmd in commands:
            rm_match = route_map_reference_pattern.search(cmd)
            if rm_match:
                route_map_name = rm_match.group(1)
                if route_map_name in route_maps:
                    new_neighbor_config.append(cmd)
            else:
                if cmd == '':
                    continue
                new_neighbor_config.append(cmd)
        bgp_neighbors[neighbor] = '\n'.join(new_neighbor_config)

    for route_map_name, route_map_config in route_maps.items():
        node = route_map_name.split('_')[1]
        neighbor = route_map_name.split('_')[-1]
        neighbor_ip = get_neighbor_ip(topo, node, neighbor)
        if not neighbor_ip or neighbor_ip not in bgp_neighbors:
            continue
        bgp_config = bgp_neighbors[neighbor_ip]
        if route_map_name not in bgp_config:
            configs = bgp_config.split('\n')
            if '!' in configs:
                configs.remove('!')
            add_config = ''
            if '_from_' in route_map_name:
                add_config = f'neighbor {neighbor_ip} route-map {route_map_name} in'
            if '_to_' in route_map_name:
                add_config = f'neighbor {neighbor_ip} route-map {route_map_name} out'
            configs.append(' ' + add_config)
            new_configs = '\n'.join(configs)
            new_configs = new_configs + '\n'
            bgp_neighbors[neighbor_ip] = new_configs
    return bgp_neighbors


def convert_json_to_bgp(json_config):
    bgp_config = ""

    # Convert JSON to Python dict if input is a JSON string
    if isinstance(json_config, str):
        json_config = json.loads(json_config)

    # Add the global BGP configuration
    bgp_config += json_config.get('bgp', '') + "\n"

    # Add each neighbor configuration
    for key, value in json_config.items():
        if key != 'bgp':
            bgp_config += value + "\n"

    # Return the final BGP configuration
    return bgp_config.strip()  # Remove any trailing newlines

def json_to_config(json_data):
    if type(json_data) == str:
        json_data = eval(json_data)

    configs = ''
    for node, config in json_data.items():
        configs = configs + f'Configuration of {node}: \n'
        configs += config.strip()
        configs += '\n\n'
    return configs


def extract_specification(text):
    # Handle both string format and direct list format
    if isinstance(text, list):
        return text
    elif isinstance(text, str):
        prefix = "Formal specifications: "
        if prefix in text:
            return eval(text.split(prefix, 1)[1])
    print("invalid specification format")
    return None


def parse_intentType(specification):
    if isinstance(specification, ECMPPathsReq):
        return 'ecmp'
    elif isinstance(specification, PathOrderReq):
        return 'order'
    elif isinstance(specification, KConnectedPathsReq):
        return 'kconnected'
    elif isinstance(specification, PathReq):
        return 'simple'
    else:
        return 'other'


def intent_to_formalspecifications(intents):
    """Transform the natural language intent into formal specification"""

    with open('./prompts/intentSpecification.txt', 'r') as f:
        prompt_template = f.read()

    prompt = prompt_template.format(Input_intents=intents)

    # print('prompt', prompt)
    specifications = ask_LLM(prompt)
    # print('specifications', specifications)
    return specifications
