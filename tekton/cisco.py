#!/usr/bin/env python
"""
Cisco specific configurations generator
Tested with IOS 15.2
"""
import itertools

from ipaddress import IPv4Interface
from ipaddress import IPv6Interface
from ipaddress import IPv4Network
from ipaddress import IPv6Network
from ipaddress import ip_network
from ipaddress import ip_address
from ipaddress import ip_interface
# from past.builtins import basestring

from tekton.graph import NetworkGraph
from tekton.bgp import Access
from tekton.bgp import ActionSetCommunity
from tekton.bgp import ActionSetLocalPref
from tekton.bgp import ActionString
from tekton.bgp import ActionASPathPrepend
from tekton.bgp import ActionSetNextHop
from tekton.bgp import ASPathList
from tekton.bgp import CommunityList
from tekton.bgp import RouteMap
from tekton.bgp import RouteMapLine
from tekton.bgp import MatchCommunitiesList
from tekton.bgp import MatchIpPrefixListList
from tekton.bgp import MatchAsPath
from tekton.bgp import MatchNextHop
from tekton.bgp import IpPrefixList

from tekton.utils import is_empty
from tekton.utils import is_symbolic


__author__ = "Ahmed El-Hassany"
__email__ = "a.hassany@gmail.com"


class CiscoConfigGen(object):
    """Generate Cisco specific configurations"""

    def __init__(self, g, prefix_map=None):
        assert isinstance(g, NetworkGraph)
        self.g = g
        self.prefix_map = prefix_map or {}
        self.nettype = (IPv4Network, IPv6Network)
        self._next_announced_prefix = int(ip_address(u'128.0.0.0'))
        self._next_as_paths = {}
        self._empty_community_lists = {}
        for node in self.g.routers_iter():
            self._next_as_paths[node] = itertools.count(1)

    def prefix_lookup(self, prefix):
        if isinstance(prefix, self.nettype):
            return prefix
        if prefix in self.prefix_map:
            return self.prefix_map[prefix]
        ip = ip_address(self._next_announced_prefix)
        net = ip_network(u"%s/24" % ip)
        self._next_announced_prefix += 256
        self.prefix_map[prefix] = net
        return net

    def gen_community_list(self, node, community_list):
        #print('gen_community_list')
        """
        Generate config lines for community list
        :param community_list:
        :return: configs string
        """
        assert isinstance(community_list, CommunityList)
        config = ''
        list_id = community_list.list_id
        # print('list_id', list_id)
        access = community_list.access.value
        communities = [c.value for c in community_list.communities if not is_empty(c)]
        if communities:
            community = f'100:{list_id-1}'
            config += "ip community-list %d %s %s\n" % (list_id, access, community)
            #config += "ip community-list %d %s %s\n" % (list_id, access, ' '.join(communities))
            #print('community_list_config', config)
        else:
            if node not in self._empty_community_lists:
                self._empty_community_lists[node] = []
            self._empty_community_lists[node].append(list_id)
        return config

    def gen_ip_prefix_list(self, node, prefix_list):
        """
        Generate config lines for ip prefix-list
        :param prefix_list:
        :return: configs string
        """
        assert isinstance(prefix_list, IpPrefixList)
        config = ''
        access = prefix_list.access.value
        networks = prefix_list.networks
        name = prefix_list.name
        for i, network in enumerate(networks):
            if is_empty(network):
                continue
            network = self.prefix_lookup(network)
            lineno = (i + 1) * 10
            network = ip_network(str(network))
            addr = str(getattr(network, 'network_address', network))
            prefixlen = getattr(network, 'prefixlen', 32)
            config += "ip prefix-list %s seq %d %s %s/%d\n" % (name, lineno,
                                                               access, addr,
                                                               prefixlen)
        return config

    def gen_as_path_list(self, node, as_path_list):
        """
        Generate config lines for AS PATH list
        :param prefix_list:
        :return: configs string
        """
        config = ''
        access = as_path_list.access.value
        as_path = as_path_list.as_paths
        name = as_path_list.list_id
        config += "ip as-path access %s %s ^%s$\n" % (name, access, '_'.join([str(t) for t in as_path[1:]]))
        return config

    def gen_iface_config(self, router, iface_name, addr, description=None, isloop=False, ospf_cost=None, acl_declare=None):
        """
        Generate configuration for a given interface
        :param iface_name: the name of the interface, e.graph. Fa0/0 or lo0
        :param addr: instance of ip_interface
        :param description: optional text description
        :return: string config
        """
        err = "Not valid address {} for iface {}:{}".format(addr, router, iface_name)
        assert isinstance(addr, (IPv4Interface, IPv6Interface)), err
        config = ''
        config += 'interface %s\n' % iface_name
        config += " ip address %s %s\n" % (addr.ip, addr.netmask)
        if acl_declare:
            for a in acl_declare:
                config += " ip access group " + a[0] + " " + a[1] + "\n"
        if ospf_cost != 'EMPTY?Value' and ospf_cost != None:
            # print('ospf_cost', ospf_cost)
            config += " ip ospf cost %d\n" % ospf_cost
            network_type = self.g.get_ospf_interface_network_type(router, iface_name)
            if network_type:
                config += " ip ospf network {}\n".format(network_type.value)
        if description:
            config += ' description "{}"\n'.format(description)
        if not isloop:
            config += " speed auto\n"
            config += " duplex auto\n"
        config += "!\n"
        return config

    def gen_all_interface_configs(self, node):
        """
        Iterate over all interfaces (including loopbacks) to generate their configs
        :param node: router name
        :return: string configs
        """
        config = ''
        for neighbor in self.g.neighbors(node):
            iface = self.g.get_edge_iface(node, neighbor)

            acl_declare = None
            acl_lists = self.g.get_acls(node)
            if acl_lists and not is_empty(acl_lists):
                if iface in acl_lists:
                    acl_declare = []
                    for acl_num in acl_lists[iface]:
                        # print("acl_num", acl_num)
                        acl_content = acl_lists[iface][acl_num]
                        acl_declare.append((str(acl_num), acl_content[0]))

            #addr = self.graph.get_edge_addr(node, neighbor)
            addr = self.g.get_iface_addr(node, iface)
            desc = self.g.get_iface_description(node, iface)
            if self.g.is_ospf_enabled(node) and self.g.is_ospf_enabled(neighbor):
                ospf_cost = self.g.get_edge_ospf_cost(node, neighbor)
            else:
                ospf_cost = None
            config += self.gen_iface_config(node, iface, addr, desc, False, ospf_cost, acl_declare)

        # Loop back interface
        for lo in sorted(self.g.get_loopback_interfaces(node)):
            addr = self.g.get_loopback_addr(node, lo)
            desc = self.g.get_loopback_description(node, lo)
            config += self.gen_iface_config(node, lo, addr, desc, True, None)
        return config

    def gen_all_communities_lists(self, node):
        """
        Get all communities list defined for the router
        :param node: router
        :return: config string
        """
        config = ""
        comm_lists = self.g.get_bgp_communities_list(node)
        #print('!!!node_comm_list', node, comm_lists)
        for num in comm_lists:
            comm_list = comm_lists[num]
            #print('num', num, 'comm_list', comm_list)
            config += self.gen_community_list(node, comm_list)
            config += "!\n"
        config += "!\n"
        return config

    def gen_all_as_path_lists(self, node):
        """
        Get all as path list defined for the router
        :param node: router
        :return: config string
        """
        config = ""
        as_path_lists = self.g.get_as_path_list(node)
        for as_path in as_path_lists.values():
            config += self.gen_as_path_list(node, as_path)
            config += "!\n"
        config += "!\n"
        return config

    def gen_all_ip_prefixes(self, node):
        """
        Generate all the ip prefixes lists
        :param node:
        :return:
        """
        config = ''
        lists = self.g.get_ip_preflix_lists(node)
        for l in lists:
            prefix_list = lists[l]
            config += self.gen_ip_prefix_list(node, prefix_list)
            config += "!\n"
        return config

    def gen_route_map_match(self, node, match):
        config = ''
        if isinstance(match, MatchCommunitiesList):
            if match.match.list_id in self._empty_community_lists.get(node, []):
                err = "Community list id {} is used in a match but it's empty".format(match.match.list_id)
                assert False, err
            config += 'match community %d' % match.match.list_id
        elif isinstance(match, MatchIpPrefixListList):
            name = match.match.name
            ips = self.g.get_ip_preflix_lists(node)
            err = "IP list '%s' is not registered at Node '%s': %s" % (name, node, ips)
            assert name in ips, err
            if not all([is_empty(p) for p in ips[match.match.name].networks]):
                config += 'match ip address prefix-list %s' % match.match.name
        elif isinstance(match, MatchAsPath):
            list_no = None
            for tmp in self.g.get_as_path_list(node).values():
                if tmp.as_paths == match.match:
                    list_no = tmp.list_id
            if not list_no:
                list_no = next(self._next_as_paths[node])
                as_path = ASPathList(list_id=list_no, access=Access.permit, as_paths=match.match)
                self.g.add_as_path_list(node, as_path)
            config += 'match as-path %s' % list_no
        elif isinstance(match, MatchNextHop):
            next_hop = match.match
            parsed = None
            if isinstance(next_hop, str):
                if '_DASH_' in next_hop:
                    parsed = next_hop.split('_DASH_')
                else:
                    parsed = next_hop.split('-')
            if parsed and self.g.has_node(parsed[0]):
                router = parsed[0]
                iface = '/'.join(parsed[1:])
                next_hop = self.g.get_interface_loop_addr(router, iface)
                self.prefix_map[match.match] = next_hop
            if hasattr(next_hop, 'ip'):
                next_hop = next_hop.ip
            config += 'match ip next-hop %s' % next_hop
        else:
            raise ValueError('Unknown match type %s' % match)
        return config

    def gen_route_map_action(self, action, list_id, peer_community):
        config = ''
        if isinstance(action, ActionSetLocalPref):
            #print('action.value', action.value)
            config += 'set local-preference %d' % int(action.value)
        elif isinstance(action, ActionSetCommunity):
            # print('action', action)
            # print('action.communities', action.communities)
            if peer_community:
                comms = peer_community
            else:
                if list_id:
                    comms = f'100:{list_id - 1}'
                else:
                    comms = ' '.join([c.value for c in action.communities])
            config += 'set community %s' % comms
            if action.additive:
                config += ' additive'
        elif isinstance(action, ActionSetNextHop):
            if '-' in action.value:
                router = action.value.split('-')[0]
                iface = '/'.join(action.value.split('-')[1:])
                addr = self.g.get_interface_loop_addr(router, iface)
                ip = addr.ip
            else:
                ip = ip_address(action.value)
            config += 'set next-hop {}'.format(str(ip))
        elif isinstance(action, ActionASPathPrepend):
            config += 'set as-path prepend %s' % ' '.join([str(x) for x in action.value])
        elif isinstance(action, ActionString):
            config = '%s' % action.value
        else:
            raise ValueError('Unknown action type %s' % action)
        return config

    def gen_route_map(self, node, routemap, peer_community):
        assert isinstance(routemap, RouteMap)
        config = ''
        name = routemap.name
        # print('node', node, 'routemap_name', name)
        # print('routemap', routemap)
        for line in routemap.lines:
            #print('!!!actions', line.actions)
            if is_empty(line.lineno) or is_empty(line.access):
                continue
            flag = False
            for action in line.actions:
                #print('_action', action)
                if isinstance(action, ActionSetLocalPref):
                    if is_empty(action.value):
                        flag = True
                elif isinstance(action, ActionSetCommunity):
                    #print('action.communities', action.communities)
                    if is_empty(action.communities[0]):
                        flag = True
            if flag:
                continue
            no = line.lineno
            access = line.access.value if hasattr(line.access, 'value') else Access.permit
            #print('access', access)
            config += "route-map %s %s %s\n" % (name, access, no)
            for match in line.matches:
                config += " %s\n" % self.gen_route_map_match(node, match)
            for action in line.actions:
                if line.matches:
                    config += " %s\n" % self.gen_route_map_action(action, line.matches[0].match.list_id, peer_community)
                else:
                    config += " %s\n" % self.gen_route_map_action(action, None, peer_community)
        return config

    def gen_all_route_maps(self, node):
        config = ''
        maps = self.g.get_route_maps(node)
        # if node in ['Velenj', 'Kranj']:
        #     print('gen_all_route_maps', node, maps)
        key = f'RMap_{node}_from_Peer{node}'
        peer_community = None
        if key in maps:
            for line in maps[key].lines:
                for action in line.actions:
                    if isinstance(action, ActionSetCommunity):
                        if not is_empty(action.communities[0]):
                            peer_community = action.communities[0].value

        for name in sorted(maps):
            routemap = maps[name]
            config += self.gen_route_map(node, routemap, peer_community)
            config += "!\n"
        return config

    def gen_bgp_config(self, node):
        """
        Generates the BGP specific configurations
        :param node: router
        :return: configs string
        """
        config = ""
        asn = self.g.get_bgp_asnum(node)
        if not asn:
            # Router doesn't have BGP configured
            return ""
        config += "router bgp %d\n" % asn
        config += ' no synchronization\n'
        config += ' bgp log-neighbor-changes\n'
        router_id = self.g.get_bgp_router_id(node)
        if router_id and not is_empty(router_id):
            if is_symbolic(router_id):
                router_id = router_id.get_value()
            config += ' bgp router-id {}\n'.format(ip_address(router_id))
        #config += ' bgp additional-paths send receive\n'
        announcements = self.g.get_bgp_announces(node)
        for ann in announcements:
            net, mask, route_map = None, None, None
            if isinstance(ann, (IPv4Network, IPv6Network)):
                net = ann.network_address
                mask = ann.netmask
            elif ann in self.g.get_loopback_interfaces(node):
                addr = self.g.get_loopback_addr(node, ann)
                net = addr.network.network_address
                mask = addr.netmask
            elif self.g.has_node(ann) and self.g.is_network(ann):
                iface = self.g.get_edge_iface(node, ann)
                addr = self.g.get_edge_iface(node, iface)
                net = addr.network.network_address
                mask = addr.netmask
            route_map = announcements[ann].get('route_map', None)
            assert net, "No network address in announcement: %s" % ann
            assert mask, "No network mask in announcement: %s" % ann
            if route_map:
                config += ' network %s mask %s route-map %s\n' % (net, mask, route_map)
            else:
                config += ' network %s mask %s\n' % (net, mask)
        for neighbor in sorted(self.g.get_bgp_neighbors(node)):
            if not self.g.is_router(neighbor):
                continue
            neighbhor_asn = self.g.get_bgp_asnum(neighbor)
            iface = self.g.get_bgp_neighbor_iface(node, neighbor)
            if is_empty(iface):
                err = "Interface for BGP peering is not defined {}->{}:{}".\
                    format(node, neighbor, iface)
                assert not is_empty(iface), err
            if iface in self.g.get_loopback_interfaces(neighbor):
                neighboraddr = self.g.get_loopback_addr(neighbor, iface)
            else:
                neighboraddr = self.g.get_iface_addr(neighbor, iface)
            assert neighbhor_asn is not None, 'AS number is not set for %s' % neighbor
            err = "BGP Peer address of {} is not set: {}.{}".format(node, neighbor, iface)
            assert neighboraddr is not None and not is_empty(neighboraddr), err
            # Check if the neighbor is peering with the loopback of this node
            source_iface = self.g.get_bgp_neighbor_iface(node, neighbor)
            update_source = source_iface in self.g.get_loopback_interfaces(node)
            config += " neighbor %s remote-as %s\n" % (neighboraddr.ip, neighbhor_asn)
            if update_source:
                config += " neighbor %s update-source %s\n" % (neighboraddr.ip, source_iface)
            description = self.g.get_bgp_neighbor_description(node, neighbor)
            if description:
                config += " neighbor %s description \"%s\"\n" % (neighboraddr.ip, description)
            config += " neighbor %s advertisement-interval 0\n" % neighboraddr.ip
            config += " neighbor %s soft-reconfiguration inbound\n" % neighboraddr.ip
            config += " neighbor %s send-community\n" % neighboraddr.ip
            import_map = self.g.get_bgp_import_route_map(node, neighbor)
            export_map = self.g.get_bgp_export_route_map(node, neighbor)
            if import_map:
                config += " neighbor %s route-map %s in\n" % (neighboraddr.ip, import_map)
            if export_map:
                config += " neighbor %s route-map %s out\n" % (neighboraddr.ip, export_map)
        return config

    def gen_tracker_configs(self, node):
        """Generate configurations for various trackers"""
        config = ""

        def get_applet(tracker_id, iface, addr, rmap, on):
            from_state, to_state = ('Down', 'Up') if on else ('Up', 'Down')
            applet_name = 'Track%s_%s' % (iface.replace('/', '_'), to_state)
            match = "%%TRACKING-5-STATE: %d interface %s line-protocol %s->%s" % (
                tracker_id, iface, from_state, to_state)
            applet = "!\n"
            applet += "event manager applet %s\n" % applet_name
            applet += ' event syslog pattern "%s"\n' % match
            applet += ' action 0.0 cli command "en"\n'
            applet += ' action 1.0 cli command "config t"\n'
            applet += ' action 2.0 cli command "router bgp %s"\n' % self.g.get_bgp_asnum(node)
            adv = 'network %s mask %s route-map %s' % (addr.network.network_address, addr.netmask, rmap)
            if on:
                applet += ' action 3.0 cli command "%s"\n' % adv
            else:
                applet += ' action 3.0 cli command "no %s"\n' % adv
            applet += ' action 4.0 cli command "end"\n'
            applet += '!\n'
            return applet

        for tracker_id in self.g.node[node].get('trackers', {}):
            data = self.g.node[node]['trackers'][tracker_id]
            iface = data['iface']
            addr = data['addr']
            rmap = data['route_map']
            config += "track %s interface %s line-protocol\n" % (tracker_id, iface)
            config += "!\n"
            config += "!\n"
            config += get_applet(tracker_id, iface, addr, rmap, True)
            config += get_applet(tracker_id, iface, addr, rmap, False)

        return config

    def gen_external_announcements(self, node):
        #assert self.g.is_peer(node)
        # Find the next free loopback interface
        next_lo = 0
        while "lo{}".format(next_lo) in self.g.get_loopback_interfaces(node):
            next_lo += 1
        ifaces = []
        lines = []
        lineno = 5
        for ann, attrs in self.g.get_bgp_advertise(node).items():
            iface = attrs.get('loopback')
            if not iface:
                net = self.prefix_lookup(ann.prefix)
                addr = ip_interface(u"%s/%d" % (next(net.hosts()), net.prefixlen))
                iface = "lo%s" % next_lo
                desc = "For %s" % ann.prefix
                self.g.set_loopback_addr(node, iface, addr)
                self.g.set_loopback_description(node, iface, desc)
            else:
                addr = self.g.get_loopback_addr(node, iface)
                net = addr.network
            # Announce network
            self.g.add_bgp_announces(node, iface)
            # Prepend AS Path
            if ann.as_path and len(ann.as_path) > 1:
                iplist = IpPrefixList(name="L_%s" % next_lo,
                                      access=Access.permit, networks=[net])
                self.g.add_ip_prefix_list(node, iplist)
                match = MatchIpPrefixListList(iplist)
                action = ActionASPathPrepend(ann.as_path)
                line = RouteMapLine(matches=[match],
                                    actions=[action],
                                    access=Access.permit,
                                    lineno=lineno)
                lines.append(line)
                ifaces.append(iface)
                lineno += 5
            next_lo += 1
        if lines:
            allow = RouteMapLine(matches=None, actions=None, access=Access.permit, lineno=100)
            lines.append(allow)
            rmap = RouteMap(name="Export_%s" % node, lines=lines)
            self.g.add_route_map(node, rmap)
            for neighbor in self.g.get_bgp_neighbors(node):
                err = "External peers cannot have predefined export route-maps (Peer %s)" % node
                assert not self.g.get_bgp_export_route_map(node, neighbor), err
                self.g.add_bgp_export_route_map(node, neighbor, rmap.name)

    def gen_all_ospf(self, node):
        if not self.g.is_ospf_enabled(node):
            return ""
        config = ""
        process_id = self.g.get_ospf_process_id(node)
        config += "router ospf %d\n" % process_id
        config += " maximum-paths 32\n"
        for network, area in self.g.get_ospf_networks(node).items():
            if network in self.g.get_loopback_interfaces(node):
                network = self.g.get_loopback_addr(node, network).network
            elif network in self.g.get_ifaces(node):
                network = self.g.get_iface_addr(node, network).network
            assert isinstance(network, (IPv4Network, IPv6Network)), "Not valid network %s" % network
            config += " network %s %s area %s\n" % (network.network_address, network.hostmask, area)
        config += "!\n"
        return config

    def gen_static_routes(self, node):
        config = ""
        static_routes = self.g.get_static_routes(node)
        if not static_routes or is_empty(static_routes):
            return config
        for prefix, next_hop in static_routes.items():
            net = self.prefix_lookup(prefix)
            if self.g.is_router(next_hop):
                iface = self.g.get_edge_iface(next_hop, node)
                addr = self.g.get_iface_addr(next_hop, iface)
                nhop_addr = addr.ip
            else:
                nhop_addr = next_hop
            config += "ip route %s %s %s\n" % (net.network_address, net.netmask, nhop_addr)
        config += "!\n"
        return config

    def gen_acl(self, node):
        config = ""
        acl_lists = self.g.get_acls(node)
        # print("acl lists", acl_lists)
        if not acl_lists or is_empty(acl_lists):
            return config
        for inface in acl_lists:
            for acl_num in acl_lists[inface]:
                acl_content = acl_lists[inface][acl_num]
                for item in acl_content[1:]:
                    # could be extended later for more acl formats
                    if len(item) == 5:
                        #item = (state, src_ip, src_mask, dst_ip, src_mask)
                        config += "access-list " + str(acl_num) + " " + str(item[0]) + " ip " + str(item[1]) + " " + str(item[2]) \
                                  + " " + str(item[3]) + " " + str(item[4]) + "\n"
                config += "access-list " + str(acl_num) + " permit any\n"
        return config


    def get_all_edge_ospf_cost(self, node):
        neighbor_cost = {}
        for neighbor in self.g.neighbors(node):
            if self.g.is_ospf_enabled(node) and self.g.is_ospf_enabled(neighbor):
                ospf_cost = self.g.get_edge_ospf_cost(node, neighbor)
            else:
                ospf_cost = None
            neighbor_cost[neighbor] = ospf_cost
        return neighbor_cost


    def gen_router_config(self, node):
            """
            Get the router configs
            :param node: router
            :return: configs string
            """
            assert self.g.is_router(node)
            #if self.g.is_peer(node):
            self.gen_external_announcements(node)

            preamble = ["version 15.2",
                        "service timestamps debug datetime msec",
                        "service timestamps log datetime msec",
                        "boot-start-marker",
                        "boot-end-marker",
                        "no aaa new-model",
                        "ip cef",
                        "no ipv6 cef",
                        "multilink bundle-name authenticated",
                        "ip forward-protocol nd",
                        "no ip http server",
                        "no ip http secure-server",
                        "ip bgp-community new-format"]
            end = ['!', '!', 'control-plane', '!', '!', 'line con 0', ' stopbits 1',
                   'line aux 0', ' stopbits 1', 'line vty 0 4', ' login', '!', 'end']

            config = ""
            # config += "\n!\nhostname {node}\n!\n".format(node=node)
            # config += "!\n"
            # config += self.gen_all_interface_configs(node)
            config += "!\n"
            config += self.gen_all_communities_lists(node)
            config += "!\n"
            # config += self.gen_all_ip_prefixes(node)
            # config += "!\n"
            if 'Peer' not in node:
                config += self.gen_all_route_maps(node)
                config += "!\n"
            # config += self.gen_bgp_config(node)
            # config += '!\n'
            # config += '!\n'
            # config += self.gen_all_as_path_lists(node)
            # config += '!\n'
            # config += "!\n"
            # config += self.gen_all_ospf(node)
            # config += "!\n"
            # config += self.gen_static_routes(node)
            # config += "!\n"
            # config += self.gen_acl(node)
            # config += "!\n"
            # config += self.gen_tracker_configs(node)
            # config += "!\n"
            # configs = "!\n" + "\n!\n".join(preamble) + config + "\n".join(end)
            # configs += "\n"
            return config
