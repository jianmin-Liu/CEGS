from utils import *

def associateNodes(user_intentGraph, example_intentGraph, gcn_model, role_text_model):
    # exact matching based on node role
    associations = {}
    for user_node, user_role in user_intentGraph['nodes'].items():
        matching_nodes = []
        for exp_node, exp_role in example_intentGraph['nodes'].items():
            if exp_role == user_role:
                matching_nodes.append(exp_node)
        associations[user_node] = matching_nodes

    # associate nodes based on neighborhood similarity
    best_associations = {}
    user_node_names = list(user_intentGraph['nodes'].keys())
    user_node_to_idx = {name: i for i, name in enumerate(user_node_names)}

    example_node_names = list(example_intentGraph['nodes'].keys())
    example_node_to_idx = {name: i for i, name in enumerate(example_node_names)}

    user_data = create_graph_data(user_intentGraph, role_text_model)
    example_data = create_graph_data(example_intentGraph, role_text_model)

    user_embedding = gcn_model(user_data.x, user_data.edge_index)
    example_embedding = gcn_model(example_data.x, example_data.edge_index)

    for user_node in user_intentGraph['nodes'].keys():
        user_node_id = user_node_to_idx[user_node]
        user_node_embedding = user_embedding[user_node_id]
        max_similarity = 0
        best_association = None
        for exp_node in associations[user_node]:
            exp_node_id = example_node_to_idx[exp_node]
            exp_node_embedding = example_embedding[exp_node_id]
            similarity = F.cosine_similarity(user_node_embedding, exp_node_embedding, dim=0)
            if similarity > max_similarity:
                max_similarity = similarity
                best_association = exp_node
        best_associations[user_node] = best_association
    return best_associations

