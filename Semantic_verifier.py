import ast
import copy
import ipaddress
import json
import math
import re
from generator import get_sub_topology, convert_configuration_to_json
from querier import ask_LLM
from netcomplete.netcomplete_ebgp_eval import *
from utils import *
from Syntax_verifier import verify_syntax
from setting_manager import setting




class local_attribute_verifier:
    def __init__(self, folder_path):
        self.folder_path = folder_path

    def parse_ospfconfig(self, configurations):
        node_interface_configs = {}
        node_ospf_configs = {}

        for node, config in configurations.items():
            interface_config = []
            ospf_config = []
            segments = config.split('!')
            # print('segments', segments)
            for seg in segments:
                # print('seg', seg)
                if seg == '':
                    continue
                if 'router ospf' in seg:
                    ospf_config.append(seg)
                elif 'interface' in seg:
                    interface_config.append(seg)
            node_interface_configs[node] = '!'.join(interface_config)
            node_ospf_configs[node] = '!'.join(ospf_config)
        return node_interface_configs, node_ospf_configs

    def parse_config_bgp(self, configurations):
        #print('config', configurations)
        #print('!!!!input', type(configurations))
        if type(configurations) == str:
            configurations = eval(configurations)
        node_interface_configs = {}
        node_bgp_configs = {}
        node_route_configs = {}
        for node in list(configurations.keys()):
            config = configurations[node]
            interface_config = []
            bgp_config = []
            route_map_config = []
            segments = config.split('!')
            # print('segments', segments)
            for seg in segments:
                # print('seg', seg)
                if seg == '':
                    continue
                if 'router bgp' in seg:
                    bgp_config.append(seg)
                elif 'interface' in seg:
                    interface_config.append(seg)
                else:
                    route_map_config.append(seg)
            if '!'.join(interface_config) != "":
                node_interface_configs[node] = '!'.join(interface_config)
            if '!'.join(bgp_config) != "":
                node_bgp_configs[node] = '!'.join(bgp_config)
            if '!'.join(route_map_config) != "":
                node_route_configs[node] = '!'.join(route_map_config)
        # print('interface_configs', node_interface_configs)
        # print('bgp_config', node_bgp_configs)
        # print('route-map config', node_route_configs)
        return node_interface_configs, node_bgp_configs, node_route_configs

    def check_ipadress(self, topo, configs):
        error = ""
        incorrect_nodes = []
        if type(configs) == 'str':
            configs = eval(configs)
        for node in configs.keys():
            node_config = configs[node]
            node_interfaces = {}
            all_interfaces_node = []

            for node_dict in topo['nodes']:
                if node_dict['name'] == node:
                    for key, value in node_dict.items():
                        if 'lo' in key:
                            node_interfaces[key] = value
                            all_interfaces_node.append(key)

            for edge in topo['edges']:
                for node_key in ['node1', 'node2']:
                    n = edge[node_key]
                    if n['name'] == node:
                        node_interfaces[n['interface']] = n['ip address']
                        all_interfaces_node.append(n['interface'])

            configed_interfaces = []
            interface_regex = re.compile(r'interface (\S+)\s+ip address (\S+) (\S+)')
            matches = interface_regex.findall(node_config)
            for match in matches:
                interface_name, ip_address, subnet_mask = match
                configed_interfaces.append(interface_name)
                cidr = f"{ip_address}/{subnet_mask.count('255') * 8}"
                if interface_name in node_interfaces:
                    if node_interfaces[interface_name] != cidr:
                        error += f'The IP configuration of {node}\'s interface {interface_name} is incorrect. '
                        error += f'The correct ip address of {node}\'s interface {interface_name} is {node_interfaces[interface_name]}.\n'
                        if node not in incorrect_nodes:
                            incorrect_nodes.append(node)
                else:
                    error += f'The {interface_name} configuration of {node}\'s is incorrect, because the interface {interface_name} does not exist in the topology. \n'
                    if node not in incorrect_nodes:
                        incorrect_nodes.append(node)

            #print('configured interfaces', configed_interfaces)
            unconfigured_interfaces = list(set(all_interfaces_node) - set(configed_interfaces))
            if len(unconfigured_interfaces) > 0:
                for interface in unconfigured_interfaces:
                    error += f'In interface configuration, {node} does not configure the interface {interface}. \n'
                    if node not in incorrect_nodes:
                        incorrect_nodes.append(node)

            # unconfigured_lookbacks = list(set(all_interfaces_node) - set(configed_interfaces))

        return error, incorrect_nodes

    def correct_interface_configs(self, nodes, configs, topology, feedback):
        prompt_paths = setting.get_prompt_paths()
        with open(prompt_paths['reflection_prefix'], 'r') as f:
            prompt_prefix = f.read()

        with open(prompt_paths['interface_suffix'], 'r') as f:
            suffix_template = f.read()

        suffix_prompt = suffix_template.format(node_set=nodes)

        prompt = prompt_prefix + f'\n\nNetwork topology: {topology}\n\n' + f'Configurations:\n{configs}\n\n' + f'Feedback:\n{feedback}\n\n' + suffix_prompt
        # print('reflect_prompt', prompt)
        reflection = ask_LLM(prompt)
        return reflection

    def verify_interface_configs(self, topo, interface_configs):
        loops = 0
        new_configurations = {}
        while True:
            if loops > 0:
                with open(f'./{self.folder_path}/interface_result.json', 'r') as file:
                    interface_configs = json.load(file)
            # print('extracted_configurations', extracted_configurations)
            if type(interface_configs) == str:
                interface_configs = eval(interface_configs)
            if len(new_configurations.keys()) > 0:
                # reflect = eval(reflect)
                for key in new_configurations.keys():
                    interface_configs[key] = new_configurations[key]
                new_configurations = {}
            # print('type', type(extracted_configurations))
            with open(f'./{self.folder_path}/interface_result.json', 'w') as json_file:
                # json_file.write(new_extracted_configurations)
                json.dump(interface_configs, json_file, indent=4)

            interface_errors, incorrect_nodes = self.check_ipadress(topo, interface_configs)

            if interface_errors != "":
                loops += 1
                subtopo = get_sub_topology(topo, incorrect_nodes)
                # print('!!!incorrect_interface_result', incorrect_result)
                incorrect_configs = {}
                for node in incorrect_nodes:
                    incorrect_configs[node] = interface_configs[node]

                configs_str = json_to_config(json.dumps(incorrect_configs))
                new_output = self.correct_interface_configs(incorrect_nodes, configs_str, subtopo, interface_errors)
                new_configurations = eval(convert_configuration_to_json(new_output))
                #print('reflection', new_output)
            else:
                break
        return loops, interface_configs

    def ip_in_network(self, ip, network, wildcard):
        # print('ip, network, wildcard', ip, network, wildcard)
        wildcard_parts = [int(part) for part in wildcard.split('.')]
        netmask_parts = [str(255 - part) for part in wildcard_parts]
        netmask = '.'.join(netmask_parts)
        network_ip = ipaddress.ip_network(f'{network}/{netmask}', strict=False)
        # print('network_ip', network_ip, type(network_ip))

        # Create IP address object
        ip = ipaddress.IPv4Network(ip)
        # print('net', network_ip, ip)
        if ip.network_address == network_ip.network_address and ip.netmask == network_ip.netmask:
            # print('ture')
            return True

        return False

    def check_networks(self, topology, ospf_config, node_name):
        ospf_processid = -1
        networks_dict = {}
        for node_dict in topology["nodes"]:
            if node_dict['name'] == node_name:
                if 'process id' in node_dict:
                    ospf_processid = node_dict['process id']
                if 'networks' in node_dict:
                    networks_dict = node_dict["networks"]
                break

        if ospf_processid == -1 and networks_dict == {}:
            return ''

        incorrect_results = ''
        configured_networks = {}
        config_lines = ospf_config.split('\n ')
        pattern_ospf_process = r'router\s+ospf\s+(\d+)'
        pattern_network = r'network\s+([0-9a-fA-F:.]+)\s+([0-9a-fA-F:.]+)\s+area\s+(\d{1,3}(?:\.\d{1,3}){0,3})'
        # pattern_network = r'(\S+)'
        for config_line in config_lines:
            # print(config_line)
            match_network = re.search(pattern_network, config_line.strip())
            match_ospf_process = re.search(pattern_ospf_process, config_line)
            if match_ospf_process:
                process_id = int(match_ospf_process.group(1))
                if process_id != ospf_processid:
                    incorrect_results += f'Node {node_name} sets incorrect ospf process id. The correct process id is {ospf_processid}.\n'
            if match_network:
                network = match_network.group(1)
                mask = match_network.group(2)
                area = int(match_network.group(3))
                #print('network', network, mask, area)
                area_flag = False
                for key, value in networks_dict.items():
                    real_area = int(key.split(' ')[1])
                    #print('area', area, 'real_area', real_area)
                    if area == real_area:
                        in_flag = False
                        #print('value', value)
                        for network_address in value:
                            if self.ip_in_network(network_address, network, mask):
                                in_flag = True
                                if f'area {area}' in list(configured_networks.keys()):
                                    configured_networks[f'area {area}'].append(network_address)
                                else:
                                    configured_networks[f'area {area}'] = [network_address]
                                break
                        if not in_flag:
                            incorrect_results += f'Node {node_name} specified an incorrect network segment {network} or network mask, because this network does not exist on the network area {real_area} of the router.\n'
                        area_flag = True
                        break
                if not area_flag:
                    incorrect_results += f'Node {node_name} specified the network segment {network} in an incorrect area {area}, because the area {area} does not exist on the router. \n'
        # print('node', node_name)
        #print('configured_networks', configured_networks)
        # unconfigured networks
        unconfigured_networks = {}
        for key, value in networks_dict.items():
            if key not in configured_networks:
                unconfigured_networks[key] = value
            else:
                unconfigured_network = list(set(value) - set(configured_networks[key]))
                if len(unconfigured_network) > 0:
                    unconfigured_networks[key] = unconfigured_network

        for key, value in unconfigured_networks.items():
            network_str = ', '.join(value)
            incorrect_results += f'Node {node_name} does not specify networks {network_str} in {key} in its OSPF configuration.\n'

        return incorrect_results

    def correct_ospf_configs(self, nodes, configs, topology, feedback):
        prompt_paths = setting.get_prompt_paths()
        with open(prompt_paths['reflection_prefix'], 'r') as f:
            prompt_prefix = f.read()

        with open(prompt_paths['ospf_suffix'], 'r') as f:
            suffix_template = f.read()

        suffix_prompt = suffix_template.format(node_set=nodes)

        prompt = prompt_prefix + f'\n\nNetwork topology: {topology}\n\n' + f'Configurations:\n{configs}\n\n' + f'Feedback:\n{feedback}\n\n' + suffix_prompt
        # print('reflect_prompt', prompt)
        reflection = ask_LLM(prompt)
        return reflection


    def verify_topology_configs_ospf(self, topo, ospf_configurations):
        loops = 0
        new_configs = {}
        while True:
            if loops > 0:
                with open(f'./{self.folder_path}/ospf_result.json', 'r') as file:
                    ospf_configurations = json.load(file)
            # print('extracted_configurations', extracted_configurations)
            if type(ospf_configurations) == str:
                ospf_configurations = eval(ospf_configurations)
            if len(new_configs.keys()) > 0:
                # reflect = eval(reflect)
                for key in new_configs.keys():
                    ospf_configurations[key] = new_configs[key]
                new_configs = {}

            with open(f'./{self.folder_path}/ospf_result.json', 'w') as json_file:
                json.dump(ospf_configurations, json_file, indent=4)

            incorrect_nodes = []
            error_results = []

            #print('ospf_configurations', ospf_configurations)

            for node, node_ospf_configs in ospf_configurations.items():
                # bgp_configs = parse_bgp_neighbors(extracted_configurations[key])
                # print('bgp_configs', bgp_configs)
                error = self.check_networks(topo, node_ospf_configs, node)
                #print('!!!ospf verfication', node, error)
                incorrect_result = f'Incorrect configuration for node {node}:\n'
                if error != '':
                    incorrect_nodes.append(node)
                    incorrect_result += error
                    error_results.append(incorrect_result)

            if len(incorrect_nodes) > 0:
                node_num = len(incorrect_nodes)
                batch_size = 50
                batches = math.ceil(node_num / batch_size)
                #print('!!!node_num', node_num, batches)
                for i in range(batches):
                    loops += 1
                    feedback = error_results[i * batch_size:min((i + 1) * batch_size, node_num)]
                    feedback = '-----------------\n'.join(feedback)
                    # print('feedback', feedback)
                    sub_nodes = incorrect_nodes[i * batch_size:min((i + 1) * batch_size, node_num)]

                    sub_node_configs = {}
                    for node in sub_nodes:
                        sub_node_configs[node] = ospf_configurations[node]

                    configs_str = json_to_config(json.dumps(sub_node_configs))
                    # nodes, configs, topology, feedback
                    subtopo = get_sub_topology(topo, sub_nodes)
                    new_output = self.correct_ospf_configs(incorrect_nodes, configs_str, subtopo, feedback)
                    sub_new_configs = eval(convert_configuration_to_json(new_output))
                    new_configs.update(sub_new_configs)
                    #print('reflection', new_configs)
            else:
                break
        return loops, ospf_configurations



    def check_neighbor(self, config, neighbors):
        # print('split', neighbor.split(' '))
        # print('check_neighbor', neighbors)
        # print('config', config)
        pattern = r'^\s*neighbor\s+\d+\.\d+\.\d+\.\d+.*$'
        match = re.findall(pattern, config, re.MULTILINE)
        unrecognized_configs = []

        if match:
            ip = config.split(' ')[1]
            # print('neighbors', neighbors)
            for nb in neighbors:
                if ip == nb['ip']:
                    return True, unrecognized_configs
            else:
                # print('!!!false')
                return False, unrecognized_configs
        else:
            unrecognized_configs.append(config)
            return False, unrecognized_configs

    def extract_networks(self, bgp_config):
        lines = bgp_config.split('\n')

        network_lines = [line.strip() for line in lines if line.strip().startswith('network')]

        return network_lines



    def check_bgp_neighbors(self, topology, bgp_config, node_name):
        # print('check_bgp_neighbors')
        nodes_AS = {}
        for node_dict in topology["nodes"]:
            name = node_dict['name']
            AS = node_dict['AS']
            nodes_AS[name] = AS

        edge_info_dict = {}
        for edge in topology["edges"]:
            node1 = edge["node1"]["name"]
            node2 = edge["node2"]["name"]
            if node1 not in edge_info_dict:
                edge_info_dict[node1] = []
            if node2 not in edge_info_dict:
                edge_info_dict[node2] = []
            value1 = {"name": node2, "ip": edge["node2"]["ip address"].split('/')[0], "AS": nodes_AS[node2]}
            if value1 not in edge_info_dict[node1]:
                edge_info_dict[node1].append(value1)
            value2 = {"name": node1, "ip": edge["node1"]["ip address"].split('/')[0], "AS": nodes_AS[node1]}
            if value2 not in edge_info_dict[node2]:
                edge_info_dict[node2].append(value2)

        incorrect_configs = {}

        # check node's AS
        incorrect_node_as = -1
        match = re.search(r"^router bgp (\d+)", bgp_config, re.IGNORECASE | re.MULTILINE)
        node_as = int(match.group(1))
        if node_as != int(nodes_AS[node_name]):
            incorrect_node_as = int(nodes_AS[node_name])
            incorrect_configs[node_name] = match.group(0)

        # print('link_dict', link_dict['Maribo'])
        pattern = r"(neighbor \d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3} remote-as \d+.*?)(?=neighbor \d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3} remote-as \d+|$)"
        # print('bgp_config', bgp_config)
        result = re.findall(pattern, bgp_config, re.DOTALL)

        # print('result', result)

        neighbor_configs = {}
        for config in result:
            # print('config', config)
            match_description = re.findall(r'neighbor (\S+) description "(.*?)"', config)
            # print('match', match_description)
            if len(match_description) > 0:
                neighbor_name = match_description[0][1].split(' ')[1]
                # print('neighbor_name', neighbor_name)
                neighbor_configs[neighbor_name] = config
            else:
                match_routemap = re.findall(r'route-map RMap_(\S+)', config)
                # print('match_routemap', match_routemap)
                if len(match_routemap) > 0:
                    neighbor_name = match_routemap[0].split('_')[-1]
                    neighbor_configs[neighbor_name] = config


        # get all neighbors
        all_neighbors = []
        for nb in edge_info_dict[node_name]:
            all_neighbors.append(nb['name'])

        incorrect_neighbors = []

        configured_neighbors = []
        not_sendcommunity_neighbors = []
        for neighbor, segment in neighbor_configs.items():
            if neighbor not in all_neighbors:
                incorrect_neighbors.append(neighbor)
                continue
            configured_neighbors.append(neighbor)
            configs = segment.split('\n ')
            sendcommunity_flag = False
            #print('configs', configs)
            for config in configs:
                if config == '\n' or config == '':
                    continue
                if 'send-community' in config:
                    sendcommunity_flag = True
                match = re.findall(r'neighbor (\S+) remote-as (\d+)', config)
                # print('match', match)
                if match:
                    # print('!!!match', match)
                    ip_address = match[0][0]
                    as_number = match[0][1]
                    # if neighbor == 'Ljublj' and node_name == 'Maribo':
                    # print('ip_address', ip_address, nb['ip'])
                    for nb in edge_info_dict[node_name]:
                        if nb['name'] == neighbor:
                            # print('neighbor', nb['name'], neighbor)
                            # print('ip_address', ip_address, nb['ip'], as_number, type(as_number), nb['AS'], type(nb['AS']))
                            if ip_address != nb['ip'] or int(as_number) != int(nb['AS']):
                                if neighbor not in incorrect_configs.keys():
                                    incorrect_configs[neighbor] = [config]
                                else:
                                    incorrect_configs[neighbor].append(config)
                else:
                    if config == '\n' or config == '':
                        continue
                    # print('not match', config)
                    pattern = r'network\s+(\d{1,3}(?:\.\d{1,3}){3})\s+mask\s+(\d{1,3}(?:\.\d{1,3}){3})'
                    network_match = re.search(pattern, config)
                    if not network_match:
                        flag, unrecognized_config = self.check_neighbor(config, edge_info_dict[node_name])
                    if not flag and len(unrecognized_config) == 0:
                        if neighbor not in incorrect_configs.keys():
                            incorrect_configs[neighbor] = [config]
                        else:
                            incorrect_configs[neighbor].append(config)
            if not sendcommunity_flag:
                not_sendcommunity_neighbors.append(neighbor)

        ## check whether some neighbors are not configured
        # print('all_neighbors', all_neighbors)
        # print('configurated_neighbors', configurated_neighbors)
        unconfigured_neighbors = list(set(all_neighbors) - set(configured_neighbors))

        # -----check network configuration----
        # get lookback ip
        networks = []

        # print('topology_routers', topology['routers'])
        for node_dict in topology['nodes']:
            # print('n', n, 'node_name', node_name)
            if node_dict['name'] == node_name:
                for key, value in node_dict.items():
                    if 'lo' in key and 'Peer' in node_name:
                        # print('!!!lookback', key, value)
                        network = ipaddress.ip_network(value, strict=False)
                        # print('network', network, network.network_address, network.subnets())
                        networks.append(network)
        # print('networks', networks)

        error_announced_networks = []
        announced_networks = []
        if len(networks) > 0:
            network_configs = self.extract_networks(bgp_config)
            for network_config in network_configs:
                pattern = r'network\s+(\d{1,3}(?:\.\d{1,3}){3})\s+mask\s+(\d{1,3}(?:\.\d{1,3}){3})'
                match = re.search(pattern, network_config)
                if match:
                    subnet = match.group(1)
                    mask = match.group(2)
                    prefix_length = sum(bin(int(x)).count('1') for x in mask.split('.'))
                    subnet_network = ipaddress.ip_network(f"{subnet}/{prefix_length}", strict=False)
                    # print('!!!subnet_network', subnet_network)
                    # print('networks', networks)
                    if subnet_network not in networks:
                        error_announced_networks.append(subnet_network)
                    else:
                        announced_networks.append(subnet_network)

        unannounced_networks = list(set(networks) - set(announced_networks))
        # if len(error_announced_networks) > 0:
        #     print('error announced networks', error_announced_networks)
        #     print('bgp config', bgp_config)
        #
        # if len(unannounced_networks) > 0:
        #     print('unannounced_networks ', unannounced_networks)
        #     print('bgp config', bgp_config)

        # print('incorrect_configs1', incorrect_configs)
        return incorrect_node_as, incorrect_configs, unconfigured_neighbors, not_sendcommunity_neighbors, unannounced_networks, error_announced_networks, incorrect_neighbors
        # print('incorrect_config', incorrect_config)


    def get_IPandAS(self, node, neighbor, topo):
        as_value = 0
        ipaddress = 0
        for node_dict in topo['nodes']:
            # print('n', n, 'neighbor', neighbor)
            if node_dict['name'] == neighbor:
                as_value = node_dict['AS']
                break
        for edge in topo['edges']:
            node1 = edge['node1']
            node2 = edge['node2']
            # print('node1', node1, 'node2', node2)
            if node1['name'] == node and node2['name'] == neighbor:
                ipaddress = node2['ip address'].split('/')[0]
            elif node2['name'] == node and node1['name'] == neighbor:
                ipaddress = node1['ip address'].split('/')[0]
        return ipaddress, as_value


    def correct_bgp_basic_configs(self, configs, nodes, topology, feedback):
        prompt_paths = setting.get_prompt_paths()
        with open(prompt_paths['reflection_prefix'], 'r') as f:
            prompt_prefix = f.read()

        with open(prompt_paths['bgp_suffix'], 'r') as f:
            suffix_template = f.read()

        suffix_prompt = suffix_template.format(node_set=nodes)

        prompt = prompt_prefix + f'\n\nNetwork topology: {topology}\n\n' + f'Configurations:\n{configs}\n\n' + f'Feedback:\n{feedback}\n\n' + suffix_prompt
        # print('reflect_prompt', prompt)
        reflection = ask_LLM(prompt)
        return reflection

    def verify_topology_configs_bgp(self, topology, bgp_configurations):
        # print('extracted_bgp_configurations', extracted_configurations)
        loops = 0
        new_configs = {}
        while True:
            if loops > 0:
                with open(f'./{self.folder_path}/bgp_result.json', 'r') as file:
                    bgp_configurations = json.load(file)
            # print('extracted_configurations', extracted_configurations)
            if type(bgp_configurations) == str:
                bgp_configurations = eval(bgp_configurations)
            if len(new_configs.keys()) > 0:
                # reflect = eval(reflect)
                for key in new_configs.keys():
                    bgp_configurations[key] = new_configs[key]
                new_configs = {}
            # print('--answer--', answer)
            with open(f'./{self.folder_path}/bgp_result.json', 'w') as json_file:
                json.dump(bgp_configurations, json_file, indent=4)

            # incorrect_result = verify_bgp_neighbor(topo, extracted_configurations)
            incorrect_configs = []
            incorrect_node = []
            error_results = []

            for node, node_bgp_config in bgp_configurations.items():
                # bgp_configs = parse_bgp_neighbors(extracted_configurations[key])
                # print('bgp_configs', bgp_configs)
                (incorrect_node_as, incorrect_config, unconfigurated_neighbors, not_sendcommunity_neighbors, unannounced_networks,
                 error_announced_networks, incorrect_neighbors) = self.check_bgp_neighbors(
                    topology, node_bgp_config, node)
                # print('!key', key, 'incorrect_config', incorrect_config)
                incorrect_result = f'Incorrect configuration for node {node}:\n'
                id = 1
                error_flag = False
                if len(list(incorrect_config.keys())) > 0:
                    error_flag = True
                    # print('!!!key', key, incorrect_config)
                    if incorrect_result != "":
                        incorrect_result += '\n'
                    for neighbor, nb_config in incorrect_config.items():
                        if neighbor == node:
                            incorrect_result += f'error-{id}:\n'
                            id += 1
                            incorrect_result += f'Node {node} is configured with an incorrect AS number in its BGP settings. The incorrect configuration is as following: \n {nb_config} \n The correct AS of this node is {incorrect_node_as}.'
                        else:
                            # print('nb_config', nb_config)
                            nb_config = '\n'.join(nb_config)
                            incorrect_result += f'error-{id}:\n'
                            id += 1
                            incorrect_result += f'Node {node} is configured with an incorrect IP address or AS or invalid configuration for its neighbor {neighbor}. The incorrect neighbor configuration is as following: \n {nb_config} \n'
                            ipaddress, as_value = self.get_IPandAS(node, neighbor, topology)
                            incorrect_result += f'The correct AS and IP address for neighbor {neighbor} are {as_value} and {ipaddress}, respectively.\n'
                    incorrect_configs.append(incorrect_config)
                if len(not_sendcommunity_neighbors) > 0:
                    error_flag = True
                    # print('!!!not_sendcommunity_neighbors', not_sendcommunity_neighbors)
                    for neighbor in not_sendcommunity_neighbors:
                        incorrect_result += f'error-{id}:\n'
                        id += 1
                        incorrect_result += f'Node {node} is not configured with a send_community command to share community information with its neighbor {neighbor}. \n'
                        ipaddress, as_value = self.get_IPandAS(node, neighbor, topology)
                        # print('ipaddress, as_value', ipaddress, as_value)
                        incorrect_result += f'\n The IP address for neighbor {neighbor} is {ipaddress}, respectively.\n\n\n'
                if len(unconfigurated_neighbors) > 0:
                    error_flag = True
                    neighbor_str = ', '.join(unconfigurated_neighbors)
                    incorrect_result += f'error-{id}:\n'
                    incorrect_result += f'Node {node} does not configure its neighbor {neighbor_str} in the BGP settings. \n'
                    id += 1
                if len(unannounced_networks) > 0:
                    error_flag = True
                    incorrect_result += f'error-{id}:\n'
                    unannounced_networks_str = []
                    for net in unannounced_networks:
                        unannounced_networks_str.append(str(net))
                    nets = ', '.join(unannounced_networks_str)
                    incorrect_result += f'\n Node {node} does set network announcement command to announce networks {nets} in the BGP configuration. \n\n'
                    id += 1
                if len(error_announced_networks) > 0:
                    error_flag = True
                    incorrect_result += f'error-{id}:\n'
                    error_announced_networks_str = []
                    for net in error_announced_networks:
                        error_announced_networks_str.append(str(net))
                    nets = ', '.join(error_announced_networks_str)
                    incorrect_result += f'\n Node {node} incorrectly announces networks {nets} in the BGP setting, because {error_announced_networks_str} do not exist in the topology. \n\n'
                    id += 1
                if len(incorrect_neighbors) > 0:
                    error_flag = True
                    neighbor_str = ', '.join(incorrect_neighbors)
                    incorrect_result += f'error-{id}:\n'
                    # The {interface_name} configuration of {node}\' is incorrect, because the interface {interface_name} does not exist in the topology
                    incorrect_result += f'\n The bgp configuration of node {node} contains the configuration regarding to nodes {neighbor_str}, but {neighbor_str} are not neighbors of {node}. \n'
                    id += 1
                if error_flag:
                    incorrect_node.append(node)
                    error_results.append(incorrect_result)

                # print('!!!incorrect', incorrect_configs, incorrect_node)
            # print('!!!incorrect_result', incorrect_node)
            # print(error_results)
            if len(incorrect_node) > 0:
                node_num = len(incorrect_node)
                batch_size = 50
                batches = math.ceil(node_num / batch_size)
                # print('!!!node_num', node_num, batches)
                for i in range(batches):
                    loops += 1
                    errors = error_results[i * batch_size:min((i + 1) * batch_size, node_num)]
                    errors = '-----------------\n'.join(errors)
                    #print('errors', errors)
                    sub_nodes = incorrect_node[i * batch_size:min((i + 1) * batch_size, node_num)]
                    subtopo = get_sub_topology(topology, sub_nodes)
                    bgp_incorrect_configs = {}
                    for node in sub_nodes:
                        bgp_incorrect_configs[node] = bgp_configurations[node]

                    bgp_incorrect_configs_str = json_to_config(bgp_incorrect_configs)
                    new_output = self.correct_bgp_basic_configs(bgp_incorrect_configs_str, sub_nodes, subtopo, errors)
                    sub_new_configs = eval(convert_configuration_to_json(new_output))
                    new_configs.update(sub_new_configs)
                    # print('bgp_reflection', reflection)
            else:
                break
        bgp_cfg = json_to_config(bgp_configurations)
        with open(f'./{self.folder_path}/bgp.txt', 'w') as f:
            f.write(str(bgp_cfg))
        return loops, bgp_configurations


class global_formal_verifier:
    def __init__(self, folder_path):
        self.folder_path = folder_path

    def parse_route_map_templates(self, config):
        """
        Parse a route-map configuration into JSON format according to the template.

        Args:
            config (str): Route-map configuration text

        Returns:
            dict: Parsed route-map in JSON format
        """
        route_map_pattern = re.compile(
            r'(^route-map .+?(?=\n!\n|\nroute-map |\ninterface |\nrouter |\n$|\Z))',
            re.DOTALL | re.MULTILINE
        )
        # Find all matches
        matches = route_map_pattern.findall(config)

        # Clean up each match (remove trailing whitespace/newlines)
        cleaned_matches = [match.strip() for match in matches]

        # print('config', config)
        # print('matches', cleaned_matches)



        route_map_templates = {}
        for segment in cleaned_matches:
            # Parse the route-map header
            header_pattern = r'route-map\s+(\S+)\s+(permit|deny)\s+(\d+)'
            header_match = re.match(header_pattern, segment.split('\n')[0].strip())
            if not header_match:
                continue
                # print('segment', segment)
                # print('cleaned_matches', cleaned_matches)
                # print('config', config)

            result = {}
            route_map_name = header_match.group(1)
            if route_map_name not in route_map_templates.keys():
                route_map_templates[route_map_name] = []
            result["name"] = route_map_name
            result["access"] = header_match.group(2)
            result["lineno"] = int(header_match.group(3))
            result["matches"] = []
            result['actions'] = []

            # Parse match and set statements
            for line in segment.split('\n')[1:]:
                # print('line', line)
                line = line.strip()
                if not line:
                    continue

                # Parse match statements
                if line.startswith('match'):
                    match_type, match_data = self.parse_match_statement(line)
                    if match_type:
                        result["matches"].append(match_data)

                # Parse set statements
                elif line.startswith('set'):
                    action = self.parse_action_statement(line)
                    if action:
                        result["actions"].append(action)

            route_map_templates[route_map_name].append(result)

        return route_map_templates

    def parse_match_statement(self, line):
        """
        Parse a match statement into the appropriate format.
        """
        if 'community' in line:
            # Example: "match community 1"
            match = re.match(r'match community\s+(\d+)', line)
            if match:
                return "MatchCommunitiesList", {
                    "match_type": "MatchCommunitiesList",
                    "communities_list": {
                        "list_id": int(match.group(1)),
                        "access": "permit", # default as defined by NetComplete
                        "communities": ["EMPTY?Value"]  # Communities would be in the community list definition
                    }
                }

        elif 'ip address prefix-list' in line:
            # Example: "match ip address prefix-list PL_EXAMPLE"
            match = re.match(r'match ip address prefix-list\s+(\S+)', line)
            if match:
                return "MatchIpPrefixListList", {
                    "match_type": "MatchIpPrefixListList",
                    "prefix_list": match.group(1)
                }

        elif 'next-hop' in line:
            # Example: "match next-hop 1.1.1.1"
            match = re.match(r'match next-hop\s+(\S+)', line)
            if match:
                return "MatchNextHop", {
                    "match_type": "MatchNextHop",
                    "next_hop": match.group(1)
                }

        return None, None

    def parse_action_statement(self, line):
        """
        Parse a set statement into the appropriate action format.
        """
        if 'local-preference' in line:
            # Example: "set local-preference 100"
            match = re.match(r'set local-preference\s+(\d+)', line)
            if match:
                return {
                    "action": "ActionSetLocalPref",
                    "value": ["EMPTY?Value"]
                }

        elif 'community' in line:
            # Example: "set community 100:0 additive"
            match = re.match(r'set community\s+([\d:]+)(?:\s+(additive))?', line)
            if match:
                return {
                    "action": "ActionSetCommunity",
                    "communities": ["EMPTY?Value"],
                    "additive": bool(match.group(2))
                }

        return None

    def is_exist(self, req, configs):
        tmp = req.split('_')
        node = tmp[2]
        neighbor_node = tmp[4]
        direction = tmp[3]
        opp_access = None
        if tmp[1] == 'Block':
            access = 'deny'
            opp_access = 'permit'
        else:
            access = 'permit'
            if direction == 'to':
                opp_access = 'deny'
        rmap1 = f'RMap_{node}_{direction}_{neighbor_node} {access}'
        rmap2 = None
        if opp_access:
            rmap2 = f'RMap_{node}_{direction}_{neighbor_node} {opp_access}'
        for node_ in configs:
            if node_ != node:
                continue
            segments = configs[node_].split('!\n')
            for seg in segments:
                if rmap2:
                    if seg.find(rmap1) and not seg.find(rmap2):
                        return True
                else:
                    if seg.find(rmap1):
                        return True
        return False

    def get_routemap_config(self, configs, node):
        routemap_config = ''
        # print('configs', configs)
        config = configs
        route_map_config = []
        segments = config.split('!')
        # print('segments', segments)
        for seg in segments:
            # print('seg', seg)
            if seg == '':
                continue
            if 'router bgp' not in seg and 'interface' not in seg:
                route_map_config.append(seg)
        routemap_config += f'Configuration of {node}:'
        routemap_config += '!'.join(route_map_config)
        routemap_config += '\n'

        return routemap_config

    def convert_errors_to_feedback(self, errors, configurations, multi_types, associations):
        configs = copy.deepcopy(configurations)
        errors = str(errors)

        # print('!!!together_flag')
        node_neighbors = []

        pattern_imp = r"\bImp_[\w]+"
        matches_imp = re.findall(pattern_imp, errors)
        imp_local_pre = {}
        for match_imp in matches_imp:
            if 'local_pref' in match_imp:
                tmp = match_imp.split('_')
                node = tmp[1]
                neighbor = tmp[3]
                if node not in imp_local_pre.keys():
                    imp_local_pre[node] = {}
                imp_local_pre[node][neighbor] = 'incorrect local preference'

        pattern_req = r"\bReq_[\w]+"
        matches_req = re.findall(pattern_req, errors)

        index_error = r"\bRmapIndexBound_[\w]+"
        matches_index = re.findall(pattern_req, errors)


        matched_nodes = []
        nodes = []
        neighbor_nodes = []
        matched_dict = {}
        feedbacks = ''
        if len(matches_req) == 0:
            if len(matches_index) > 0:
                for match in matches_index:
                    node = match.split('_')[2]
                    if node not in nodes:
                        nodes.append(node)
            return errors, nodes
        matches_req_ = []

        for req in matches_req:
            # flag = self.is_exist(req, configs)
            # if flag:
            #     continue
            matches_req_.append(req)
            node = req.split('_')[2]
            neighbor_node = req.split('_')[4]
            node_neighbors.append((node, neighbor_node))

            nodes.append(node)
            neighbor_nodes.append(neighbor_node)

            if not multi_types:
                matched_node = associations[0][node]

                tmp = req.split('_')

                req_desc = ' '.join(tmp[1:5])
                if tmp[1] == 'Block':
                    req_desc = f'{tmp[2]} Denies traffic {tmp[3]} {tmp[4]}'
                elif tmp[1] == 'Allow':
                    req_desc = f'{tmp[2]} Permits traffic {tmp[3]} {tmp[4]}'
                feedback = (f'Request \'{req_desc}\' is not satisfied. The configurations of node {node} may be incorrect. Infer and correct the route-map configurations of node {node} regarding its neighbor {neighbor_node}, referring to the configuration of its matching node {matched_node} in example topology T2. '
                            f'\n') #  Note: if you specify a route-map with permit action, you must give the corresponding match and set statement.
                feedbacks += feedback
            else:
                matched_nodes = []
                for association in associations:
                    matched_nodes = association[node]

                tmp = req.split('_')

                req_desc = ' '.join(tmp[1:5])
                if tmp[1] == 'Block':
                    req_desc = f'{tmp[2]} Denies traffic {tmp[3]} {tmp[4]}'
                elif tmp[1] == 'Allow':
                    req_desc = f'{tmp[2]} Permits traffic {tmp[3]} {tmp[4]}'
                feedback = f'Request \'{req_desc}\' is not satisfied. The configurations of node {node} may be incorrect. Infer and correct the route-map configurations of node {node} regarding its neighbor {neighbor_node}, referring to the configuration of its matching nodes {matched_nodes} in example topology T2.\n'
                feedbacks += feedback
            # print('feed', feedback)
        # print('!!!feed', feedbacks)
        nodes = list(set(nodes))
        nodes_ = nodes

        if feedbacks == '':
            if len(matches_req_) > 0:
                feedbacks = f'Requests {matches_req_} are not satisfied. Correct the route-map configurations of {set(nodes_)} regarding their neighbors, referring to the route-map configurations of its matching node in example Topology T2.'
            else:
                feedbacks = f'{errors}\n Reflect on which routers have incorrect configurations and correct them according to the configuration example.'

        return feedbacks, nodes_

    def get_routemap_configs(self, configs, node):
        routemap_config = ''
        # print('configs', configs)
        config = configs
        route_map_config = []
        segments = config.split('!')
        # print('segments', segments)
        for seg in segments:
            # print('seg', seg)
            if seg == '':
                continue
            if 'router bgp' not in seg and 'interface' not in seg:
                route_map_config.append(seg)
        routemap_config += f'Configuration of {node}:'
        routemap_config += '!'.join(route_map_config)
        routemap_config += '\n'

        return routemap_config

    def verify_dst_config(self, dst, dst_config, topology, associations):
        neighbors = []
        for edge in topology['edges']:
            node1 = edge['node1']['name']
            node2 = edge['node2']['name']
            if node1 == dst:
                neighbors.append(node2)
            elif node2 == dst:
                neighbors.append(node1)
        feedback = ''
        matched_node = associations[0][dst]
        for neighbor in neighbors:
            route_map = f'route-map RMap_{dst}_to_{neighbor} permit'
            if route_map not in dst_config:
                feedback += (f'Request {dst} Permits traffic to {neighbor} is not satisfied. The configurations of node {dst} is incorrect. Infer and correct the route-map configurations of node {dst} regarding its neighbor {neighbor}, referring to the configuration of its matching node {matched_node} in example topology T2.'
                             f' Note: you need to specify the set community statement. \n')
            else:
                segments = dst_config.split('!\n')
                #print('segs', segments)
                for seg in segments:
                    if route_map in seg:
                        if 'set community' not in seg:
                            feedback += (f'The route-map configuration of {dst}_to_{neighbor} is incorrect because it lacks the set community statement.'
                                         f' Infer and correct the route-map configurations of node {dst} regarding its neighbor {neighbor}, referring to the configuration of its matching node {matched_node} in example topology T2.\n')
                        break
        return feedback



    def verify_bgp_routing_policy_dst(self, destination, target_topology, dst_config, associations, configurationExample, intents):
        loops = 0
        while True:
            feedback = self.verify_dst_config(destination, dst_config, target_topology, associations)
            #print('!!!correct_dst_config', feedback)
            if feedback == '':
                break

            loops += 1
            configs = {}
            configs[destination] = dst_config
            reflection = self.correct_bgp_routing_policy_configs(intents, target_topology, configs,
                                                                 [destination], configurationExample, associations,
                                                                 feedback)
            #print('reflection', reflection)
            des_tmp_configs = eval(convert_configuration_to_json(reflection))
            dst_config = des_tmp_configs[destination]
            #print('new dst', dst_config)
        return dst_config, loops

    def verify_src_config(self, src, relays, src_config, associations):
        feedback = ''
        matched_node = associations[0][src]
        for relay in relays:
            route_map = f'route-map RMap_{src}_from_{relay} permit'
            segments = src_config.split('!\n')
            is_exist = False
            for seg in segments:
                if route_map in seg:
                    is_exist = True
                    if 'match community' not in seg:
                        feedback += (
                            f'The route-map configuration of {src}_from_{relay} is incorrect because it lacks the match community statement.'
                            f' Infer and correct the route-map configurations of node {src} regarding its neighbor {relay}, referring to the configuration of its matching node {matched_node} in example topology T2.\n')
                    break
            if not is_exist:
                feedback += (f'Request {src} Permits traffic from {relay} is not satisfied. The configurations of node {src} is incorrect. Infer and correct the route-map configurations of node {src} regarding its neighbor {relay}, referring to the configuration of its matching node {matched_node} in example topology T2.'
                             f' Note: you need to specify the match community statement. \n')

        return feedback


    def verify_bgp_routing_policy_src(self, src, paths, target_topology, src_config, associations, configurationExample, intents):
        relays = []
        for path in paths:
            relays.append(path[1])
        loops = 0
        while True:
            feedback = self.verify_src_config(src, relays, src_config, associations)
            #print('!!!correct_src_config', feedback)
            if feedback == '':
                break

            loops += 1
            configs = {}
            configs[src] = src_config
            reflection = self.correct_bgp_routing_policy_configs(intents, target_topology, configs,
                                                                 [src], configurationExample, associations,
                                                                 feedback)
            #print('reflection', reflection)
            src_tmp_configs = eval(convert_configuration_to_json(reflection))
            src_config = src_tmp_configs[src]
            #print('new src', src_config)
        return src_config, loops



    def correct_bgp_routing_policy_configs(self, intents, topology, configs, incorrect_nodes, configurationExample, associations, feedback):
        prompt_paths = setting.get_prompt_paths()
        with open(prompt_paths['bgp_routing_policy'], 'r') as f:
            prompt_template = f.read()

        with open(prompt_paths['routemap_format'], 'r') as f:
            configuration_format = f.read()

        if len(intents) == 1:
            user_intent = intents[0]
        else:
            user_intent = ""
            intent_id = 1
            for intent in intents:
                user_intent += f'Intent{intent_id}: {intent}\n'
                intent_id += 1

        if len(intents) == 1:
             multi_type = False
        else:
            multi_type = True

        subtopo = get_sub_topology(topology, incorrect_nodes)
        incorrect_configs = {}
        sub_associations = {}
        matched_nodes = []
        example_configs = ''
        if not multi_type:
            for node in incorrect_nodes:
                incorrect_configs[node] = configs[node]
                sub_associations[node] = associations[0][node]

            for exp_node, exp_config in configurationExample[0]['configurations'].items():
                example_configs = example_configs + self.get_routemap_configs(exp_config, exp_node)

            example_topology = configurationExample[0]['topology']
            example_intent = configurationExample[0]['intent']
            examples = f'Example topology:\n{example_topology}\n\n' + f'Example intent:\n{example_intent}\n\n' + f'Configurations:\n{example_configs}'
        else:
            #example_configs = json_to_config(configurationExample['configurations'])
            examples = ''
            all_example_configs = {}
            for node in incorrect_nodes:
                incorrect_configs[node] = configs[node]
                if node not in sub_associations:
                    sub_associations[node] = []

                for i in range(len(associations)):
                    # if f'intent{i+1}' not in all_example_configs.keys():
                    #     all_example_configs[f'intent{i+1}'] = ''
                    assoc = associations[i]
                    sub_associations[node].append(assoc[node])
                #     config_exp = configurationExample[i]
                #     all_example_configs[f'intent{i+1}'] = all_example_configs[f'intent{i+1}'] + self.get_routemap_configs(
                #         config_exp['configurations'][assoc[node]], assoc[node])
            for i in range(len(configurationExample)):
                all_example_configs[f'intent{i + 1}'] = ''
                for exp_node, exp_config in configurationExample[i]['configurations'].items():
                    all_example_configs[f'intent{i + 1}'] = all_example_configs[f'intent{i + 1}'] + self.get_routemap_configs(
                        exp_config, exp_node)
            for i in range(len(associations)):
                example_topology = configurationExample[i]['topology']
                example_intent = configurationExample[i]['intent']
                example_config = all_example_configs[f'intent{i+1}']
                examples = examples + f'***\nExample topology:\n{example_topology}\n\n' + f'Example intent:\n{example_intent}\n\n' + f'Configurations:\n{example_config}\n****\n'


        incorrect_configs_description = json_to_config(incorrect_configs)

        #print('incorrect_configs', incorrect_configs_description)

        prompt = prompt_template.format(examples=examples, target_topology=subtopo,
                                 user_intent=user_intent, incorrect_configurations=incorrect_configs_description,
                                 feedback=feedback, associations=sub_associations, incorrect_nodes=incorrect_nodes,
                                 configuration_format=configuration_format)
        #print('prompt', prompt)

        return ask_LLM(prompt)


    def verify_bgp_routing_policy(self, intent_id, intents, target_topology, routing_policy_configs, associations, configurationExample):
        # if type(extracted_configurations) == str:
        #     extracted_configurations = eval(extracted_configurations)

        if len(intents) == 1:
             multi_type = False
        else:
            multi_type = True


        formal_intents = []
        reqsize = 0
        intent_types = []
        intent_type = None
        for intent in intents:
            reqsize += 1
            user_specification_intent = intent_to_formalspecifications(intent)
            #print('user_specification_intent', user_specification_intent)
            user_specification_list = extract_specification(user_specification_intent)
            formal_intents.append(user_specification_list[0])
            #print('formal_intents', formal_intents)
            intent_type = parse_intentType(user_specification_list[0])
            # isinstance(protocol, Protocols)
            if intent_type == 'other':
                raise ValueError("Unsupported intent type verification %s", intent_type)
            if len(intent_types) == 0:
                intent_types.append(intent_type)
            else:
                if intent_type not in intent_types:
                    intent_types.append(intent_type)
                    raise ValueError("Unsupported intent type verification %s", ', '.join(intent_types))

        dst_loops = 0
        src_loops = 0
        if not multi_type:
            if intent_type == 'order':
                source = formal_intents[0].paths[0].path[0]
                paths = [formal_intents[0].paths[0].path, formal_intents[0].paths[1].path]
                routing_policy_configs[source], src_loops = self.verify_bgp_routing_policy_src(source, paths, target_topology, routing_policy_configs[source], associations, configurationExample, intents)
            destination = formal_intents[0].dst_net
            routing_policy_configs[destination], dst_loops = self.verify_bgp_routing_policy_dst(destination, target_topology, routing_policy_configs[destination], associations, configurationExample, intents)

        # print('!!!formal_intents', formal_intents)
        #print('intent_type', intent_types)

        new_configs = {}
        loops = 0
        total_time = 0
        while True:
            #print('________________loops_________________', loops)
            if loops > 0:
                # config_file_path = folder_path + '/response' + str(intent_id) + '.txt'
                # with open(config_file_path, 'r', encoding='utf-8') as file:
                # print('from__', '.json')
                file_path = self.folder_path + '/route_map_results/result' + str(intent_id) + '.json'
                with open(file_path, 'r') as file_:
                    routing_policy_configs = json.load(file_)

            if len(new_configs.keys()) > 0:
                # reflect = eval(reflect)
                for key in new_configs.keys():
                    routing_policy_configs[key] = new_configs[key]

            file_path = self.folder_path + '/route_map_results/result' + str(intent_id) + '.json'
            with open(file_path, 'w') as json_file:
                json.dump(routing_policy_configs, json_file, indent=4)

            #print('routing_policy_configs', routing_policy_configs)


            templates = {}
            templates['rmaps'] = {}
            # print('configs', configs)
            # print('extracted_configurations_type', type(extracted_configurations))
            if type(routing_policy_configs) == str:
                routing_policy_configs = eval(routing_policy_configs)
            for node, node_route_map_configs in routing_policy_configs.items():
                route_map_template = self.parse_route_map_templates(node_route_map_configs)
                #print('route_map_templates', route_map_template)
                for key, tem in route_map_template.items():
                    templates['rmaps'][key] = tem

            template_file = 'routing_policy_templates.json'
            with open(template_file, 'w') as json_file:
                json.dump(templates, json_file, indent=4)

            #print('formal intents', formal_intents)

            errors, synthesized_routeMap_configs = Bgpeval(intent_types[0], reqsize, formal_intents, template_file)
            #print('!!!errors', errors)

            if errors:
                loops += 1
                feedback, incorrect_nodes = self.convert_errors_to_feedback(errors, routing_policy_configs, multi_type, associations)
                #print('feedback', feedback)
                reflection = self.correct_bgp_routing_policy_configs(intents, target_topology, routing_policy_configs, incorrect_nodes, configurationExample, associations, feedback)
                #print('reflection', reflection)
                new_configs = eval(convert_configuration_to_json(reflection))
                #print('new_configs', new_configs)
            else:
                #print('!!!successfully')
                with open(f'{self.folder_path}/configurations/route_map_configs.json', 'w') as json_file:
                    json.dump(synthesized_routeMap_configs, json_file, indent=4)
                break
            if loops > 300:
                break
        return loops + dst_loops + src_loops

def verify_bgp(GFV, LAV, intent_id, user_intent, topology, configurations, associations, configurationExample):
    print('=====verify BGP configurations=====')
    #interface_config_json, bgp_config_json, route_map_config_json = LAV.parse_config_bgp(configurations)

    syntax_loops = 0

    syntax_loops, configurations = verify_syntax(configurations, topology)
    interface_config_json, bgp_config_json, route_map_config_json = LAV.parse_config_bgp(configurations)

    # print('!!!interface_config_json', json.dumps(interface_config_json, indent=4))
    #print('!!!bgp_configs', bgp_config_json)
    interface_loops = 0
    topo_loops = 0

    if not os.path.exists(f'./{LAV.folder_path}/configurations/interface.json'):
        merge_interface_configs = {}
        nodes = []
        for node in interface_config_json.keys():
            nodes.append(node)

        batch_size = 300
        batches = math.ceil(len(nodes) / batch_size)
        for k in range(batches):
            subnodes = nodes[k * batch_size:min((k + 1) * batch_size, len(nodes))]
            subtopo = get_sub_topology(topology, subnodes)
            #print('!!!subtopo routers111', subtopo['nodes'])
            sub_interface_configs = {}

            for node in subnodes:
                sub_interface_configs[node] = interface_config_json[node]


            interface_loop, sub_interface_configs = LAV.verify_interface_configs(subtopo, sub_interface_configs)
            interface_loops += interface_loop
            merge_interface_configs.update(sub_interface_configs)
        with open(f'./{GFV.folder_path}/configurations/interface.json', 'w') as json_file:
            json.dump(merge_interface_configs, json_file, indent=4)

    if not os.path.exists(f'./{LAV.folder_path}/configurations/bgp.json'):
        merge_bgp_configs = {}
        nodes = []
        for node in bgp_config_json.keys():
            nodes.append(node)

        batch_size = 300
        batches = math.ceil(len(nodes) / batch_size)

        for k in range(batches):
            subnodes = nodes[k * batch_size:min((k + 1) * batch_size, len(nodes))]
            subtopo = get_sub_topology(topology, subnodes)
            sub_bgp_configs = {}
            for node in subnodes:
                sub_bgp_configs[node] = bgp_config_json[node]

            topo_loop, sub_bgp_configs = LAV.verify_topology_configs_bgp(subtopo, sub_bgp_configs)
            topo_loops += topo_loop
            merge_bgp_configs.update(sub_bgp_configs)

        with open(f'./{GFV.folder_path}/configurations/bgp.json', 'w') as json_file:
            json.dump(merge_bgp_configs, json_file, indent=4)

    print('====verify routing policy configurations====')
    policy_loops = GFV.verify_bgp_routing_policy(intent_id, [user_intent], topology, route_map_config_json, [associations],
                             [configurationExample])

    total_loops = syntax_loops + interface_loops + topo_loops + policy_loops
    print('====loops for correcting routing policy configurations', policy_loops)

    #print(f'intent {intent_id} loops', (total_loops))

    return total_loops



def verify_ospf(LAV, topology, configurations):
    print('====verify OSPF configuration====')
    syntax_loops, configurations = verify_syntax(configurations, topology)
    interface_config_json, ospf_config_json = LAV.parse_ospfconfig(configurations)
    # print('!!!interface_config_json', interface_config_json)
    interface_loops = 0
    topo_loops = 0

    nodes = []
    for node_dict in topology['nodes']:
        nodes.append(node_dict['name'])

    merge_interface_configs = {}
    merge_ospf_configs = {}
    batch_size = 300
    parts = math.ceil(len(nodes) / batch_size)
    for k in range(parts):
        subnodes = nodes[k * batch_size:min((k + 1) * batch_size, len(nodes))]
        subtopo = get_sub_topology(topology, subnodes)
        # print('!!!subtopo routers111', subtopo['routers'])
        sub_interface_configs = {}
        sub_ospf_configs = {}
        for node in subnodes:
            if node in interface_config_json.keys():
                sub_interface_configs[node] = interface_config_json[node]
            if node in ospf_config_json.keys():
                sub_ospf_configs[node] = ospf_config_json[node]

        # print('!!!subtopo routers', subtopo['routers'])
        interface_loop, sub_interface_configs = LAV.verify_interface_configs(subtopo, sub_interface_configs)

        #print('fish verify_interface_configs')
        interface_loops += interface_loop
        merge_interface_configs.update(sub_interface_configs)
        topo_loop, sub_ospf_configs = LAV.verify_topology_configs_ospf(subtopo, sub_ospf_configs)
        #print('topo_loop', topo_loop)
        topo_loops += topo_loop
        merge_ospf_configs.update(sub_ospf_configs)

    print('====loops for correcting interface configurations====', interface_loops)
    print('====loops for correcting OSPF basic configurations====', topo_loops)
    with open(f'./{LAV.folder_path}/configurations/interface.json', 'w') as json_file:
        json.dump(merge_interface_configs, json_file, indent=4)
    with open(f'./{LAV.folder_path}/configurations/ospf.json', 'w') as json_file:
        json.dump(merge_ospf_configs, json_file, indent=4)

    # smt_time = generate_edge_cost(req_num, intent, folder_path, topo, prefix_prompt, False)

    return syntax_loops + interface_loops + topo_loops, 0



def verify_configurations(protocol, LAV, GFV, node_configurations, intent_id, user_intent, target_topo, associations, configurationExample):
    total_loops = 0
    smt_time = 0
    if protocol == 'OSPF':
        loops, _ = verify_ospf(LAV, target_topo, node_configurations)
        total_loops += loops
    elif protocol == 'Static':
        loops = 0
        print()
    elif protocol == 'BGP':
        print('!!!verify bgp')
        # configured_nodes, topo, intent, prefix_prompt, config_json, req_num, folder_path, node_match_dict
        loops = verify_bgp(GFV, LAV, intent_id, user_intent, target_topo, node_configurations, associations, configurationExample)
        interface_config_flag = True
        bgp_config_flag = True
        #SMT_solve_time += smt_time
        total_loops += loops

    return total_loops