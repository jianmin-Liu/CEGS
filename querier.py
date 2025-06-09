import json
from sklearn.metrics.pairwise import cosine_similarity
from utils import *
from gensim.models import KeyedVectors
from setting_manager import setting

class FastTextFeatureExtractor:
    def __init__(self, model_name='fasttext-wiki-news-subwords-300'):
        """
        initialize FastText feature extractor
        model_name: pre-trained model name:
        - 'fasttext-wiki-news-subwords-300' (300d)
        - 'word2vec-google-news-300' (300d)
        """
        try:
            #self.model = api.load(model_name)
            self.model = KeyedVectors.load_word2vec_format('wiki-news-300d-1M-subword.vec', binary=False)
            self.embedding_dim = self.model.vector_size
            print(f"FastText model loaded successfully, dimension: {self.embedding_dim}")
        except Exception as e:
            self.model = None
            self.embedding_dim = 300
            self._create_simple_embeddings()

    def _create_simple_embeddings(self):
        """create simple vocabulary and random embeddings"""
        self.vocab = {}
        self.embeddings = {}
        self.embedding_dim = 300

    def _get_word_vector(self, word):
        """get word vector representation"""
        if self.model is not None:
            try:
                return self.model[word]
            except KeyError:
                return np.zeros(self.embedding_dim)
        else:
            if word not in self.embeddings:
                np.random.seed(hash(word) % (2 ** 32))
                self.embeddings[word] = np.random.normal(0, 0.1, self.embedding_dim)
            return self.embeddings[word]

    def encode_text(self, text):
        """encode text to vector"""
        if isinstance(text, str):
            words = text.lower().replace('-', ' ').replace('_', ' ').split()
            if not words:
                return np.zeros(self.embedding_dim)

            vectors = []
            for word in words:
                vector = self._get_word_vector(word)
                vectors.append(vector)

            return np.mean(vectors, axis=0)
        else:
            return np.zeros(self.embedding_dim)

    def encode_texts(self, texts):
        """batch encode texts"""
        return np.array([self.encode_text(text) for text in texts])
    
    def encode(self, text):
        """uniform interface: encode single text"""
        return self.encode_text(text)


def process_intent(user_intent):
    """process user intent"""
    prompt_paths = setting.get_prompt_paths()
    with open(prompt_paths['intent_process'], 'r') as f:
        prefix_prompt = f.read()

    prompt = prefix_prompt + f'\n\nIntent:\n{user_intent}'
    intent = ask_LLM(prompt)
    return intent



def queryExamplewithSimilarIntent(user_intent):
    """query examples with similar intent"""
    data_paths = setting.get_data_paths()
    model_paths = setting.get_model_paths()
    
    with open(data_paths['example_library'], 'r', encoding='utf-8') as file:
        example_library = json.load(file)

    threshold = 0.9
    topk_examples = []
    examples = []
    for exp_id in example_library.keys():
        example = example_library[exp_id]
        example_intent = example['normalized intent']
        model = SentenceTransformer(model_paths['sentence_transformer'])
        embeddings = model.encode([example_intent, user_intent])
    
        cosine_sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        if cosine_sim > threshold:
            topk_examples.append(example)
            examples.append(exp_id)
    
    print('top-k examples', examples)
    return topk_examples


def extract_content_with_angle_brackets(text):
    """extract content with angle brackets"""
    pattern = r'<(.*?)>'
    matches = re.findall(pattern, text)
    paths = []
    for match in matches:
        nodes = match.split(', ')
        paths.append(nodes)
    return paths


def matching_node(graph1, graph2, gcn_model, text_model):
    """matching nodes between two graphs"""
    g1_data = create_graph_data(graph1, text_model)
    g2_data = create_graph_data(graph2, text_model)
    print('g1_data', g1_data)
    sim_avg = 0
    num = 0

    gcn_model.eval()
    with torch.no_grad():
        g1_embedding = gcn_model(g1_data.x, g1_data.edge_index)
        g2_embedding = gcn_model(g2_data.x, g2_data.edge_index)

        num_nodes_topo1 = g1_embedding.shape[0]
        num_nodes_topo2 = g2_embedding.shape[0]

        for i in range(num_nodes_topo1):
            num += 1
            min_dist = 10000000
            for j in range(num_nodes_topo2):
                dist = F.pairwise_distance(g1_embedding[i].unsqueeze(0), g2_embedding[j].unsqueeze(0), p=1)
                if dist < min_dist:
                    min_dist = dist
            sim_avg += min_dist
        sim_avg /= num
    return sim_avg

def calculateGraphSim(graph1, graph2):
    "Calculate the similarity between two intent graphs"

    text_model = FastTextFeatureExtractor() 

    model_paths = setting.get_model_paths()
    gcn_model = GraphSAGE(in_channels=300, hidden_size=256, out_channels=128, num_layers=2)
    
    checkpoint = torch.load(model_paths['gcn'], map_location=torch.device('cpu'))
    gcn_model.load_state_dict(checkpoint['gcn_state_dict'])
    
    g1_data = create_graph_data(graph1, text_model)
    g2_data = create_graph_data(graph2, text_model)
    
    gcn_model.eval()
    with torch.no_grad():
        g1_embedding = gcn_model(g1_data.x, g1_data.edge_index)
        g2_embedding = gcn_model(g2_data.x, g2_data.edge_index)

        g1_graph_embed = torch.mean(g1_embedding, dim=0)  # [embed_dim]
        g2_graph_embed = torch.mean(g2_embedding, dim=0)  # [embed_dim]

    dist = F.pairwise_distance(g1_graph_embed.unsqueeze(0),
                                   g2_graph_embed.unsqueeze(0), p=1)
    return 1 / (1 + dist.item())


def get_intenttype(intent):
    data_paths = setting.get_data_paths()
    model_paths = setting.get_model_paths()
    
    with open(data_paths['intent_types'], 'r') as f:
        types = json.load(f)

    max_similarity = 0
    optimal_type = None
    for type, defination in types.items():
        model = SentenceTransformer(model_paths['sentence_transformer'])
        embeddings = model.encode([intent, defination])
        cosine_sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        if cosine_sim > max_similarity:
            optimal_type = type
            max_similarity = cosine_sim
    if max_similarity > 0.8:
        return optimal_type
    else:
        return 'other'


def retrievalExamples(user_intent, normalized_user_intent, target_topology):
    """
    Identify the relevant configuration example
    """
    topk_examples = queryExamplewithSimilarIntent(normalized_user_intent)
    intent_type = get_intenttype(normalized_user_intent)
    user_graph = generateGraph(intent_type, user_intent, target_topology)
    best_example = None
    max_similarity = 0
    for example in topk_examples:
        graph = example['intent_graph']
        similarity = calculateGraphSim(graph, user_graph)
        if similarity > max_similarity:
            max_similarity = similarity
            best_example = example

    return best_example, user_graph, best_example['intent_graph']