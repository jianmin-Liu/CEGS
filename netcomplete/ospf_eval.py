#!/usr/bin/env python

import argparse
import json
import logging
import random
import sys
import os

from timeit import default_timer as timer
from synet.utils.topo_gen import read_topology_zoo_netgraph
from synet.utils.smt_context import VALUENOTSET
from synet.synthesis.ospf_heuristic import OSPFSyn as OSPFCEGIS
from synet.synthesis.ospf import OSPFSyn as OSPFConcrete
from synet.synthesis.connected import ConnectedSyn
from synet.utils.common import PathReq
from synet.utils.common import ECMPPathsReq
from synet.utils.common import PathOrderReq
from synet.utils.common import Protocols
from synet.utils.common import KConnectedPathsReq
from synet.utils.topo_gen import read_topology_from_json


def setup_logging():
    # create logger
    logger = logging.getLogger('synet')
    logger.setLevel(logging.DEBUG)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)


def ospfeval(req_type, formal_intents):
    reqsize = len(formal_intents)
    fixed = 0
    seed = 0
    k = 2
    syn = 'cegis'
    path_gen = 100

    assert 0 <= fixed <= 1.0

    # Generate new random number seed if need
    if not seed:
        seed = random.randint(0, sys.maxsize)

    # This random generator MUST be used everywhere!!!!
    ospfRand = random.Random(seed)

    topo_file = './input/topology.json'
    with open(topo_file, 'r') as json_file:
        topo_json_file = json.load(json_file)
        # print('topo_json_file', topo_json_file)

    topo = read_topology_from_json(topo_json_file)

    vals = []
    if req_type == 'simple':
        reqs = formal_intents
    elif req_type == 'ecmp':
        reqs = formal_intents
    elif req_type == 'kconnected':
        reqs = formal_intents
    elif req_type == 'order':
        reqs = formal_intents
    else:
        raise ValueError("Unknow req type %s", req_type)

    for node in topo.nodes():
        topo.enable_ospf(node, 100)
    # Initially all costs are empty
    for src, dst in topo.edges():
        topo.set_edge_ospf_cost(src, dst, VALUENOTSET)
    # how many is fixed
    fixed_edges = ospfRand.sample(vals, int(round(len(vals) * fixed)))
    for src, dst, cost in fixed_edges:
        #print "Fixing", src, dst, cost
        topo.set_edge_ospf_cost(src, dst, cost)
    # Establish basic connectivity
    conn = ConnectedSyn([], topo, full=True)
    conn.synthesize()

    t1 = timer()
    if syn == 'cegis':
        ospf = OSPFCEGIS(topo, gen_paths=path_gen, random_obj=ospfRand)
        for req in reqs:
            ospf.add_req(req)
        assert ospf.synthesize()
        assert not ospf.removed_reqs
    elif syn == "concrete":
        ospf = OSPFConcrete(topo)
        for req in reqs:
            ospf.add_req(req)
        assert ospf.solve()
    else:
        raise ValueError("Unknow syn type %s" % syn)


    from tekton.gns3 import GNS3Topo
    ospf.update_network_graph()
    gns3 = GNS3Topo(topo)
    edge_cost = gns3.get_edge_cost()

    return edge_cost
