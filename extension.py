from gensim.models import Word2Vec
import networkx as nx
from rdflib import Graph, Namespace
import random
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# 1. Load RDF Data
def load_rdf_data():
    graphs = []
    try:
        g1 = Graph()
        g1.parse(
            "/Users/yingji/Library/CloudStorage/OneDrive-MicrosoftOffice365/VU_course/AI_master/ML for Graph/RDF2Vec/dataset/AIFB.n3",
            format="n3")
        graphs.append(g1)
    except Exception as e:
        print(f"Error parsing RDF files: {e}")
    return graphs


# 2. Generate Graph Walks (BFS with Centrality-Based Weights)
def generate_graph_walks(graphs, depth=10):
    walks = []
    for graph in graphs:
        nx_graph = nx.DiGraph()
        for subj, pred, obj in graph:
            nx_graph.add_edge(str(subj), str(obj), label=str(pred))

        centrality = nx.degree_centrality(nx_graph)

        visited = set()
        for node in nx_graph.nodes:
            if node not in visited:
                queue = [(node, [node])]
                visited.add(node)

                for _ in range(depth):
                    next_queue = []
                    for current, path in queue:
                        neighbors = list(nx_graph.neighbors(current))
                        if not neighbors and len(path) == 1:
                            walks.append(path)

                        # Weighted random selection based on centrality
                        if neighbors:
                            weights = [centrality.get(neighbor, 0.01) for neighbor in neighbors]
                            total_weight = sum(weights)
                            probabilities = [w / total_weight for w in weights]

                            selected_neighbor = random.choices(neighbors, weights=probabilities, k=1)[0]
                            new_path = path + [selected_neighbor]
                            next_queue.append((selected_neighbor, new_path))

                            if len(new_path) == depth:
                                walks.append(new_path)
                            visited.add(selected_neighbor)
                    queue = next_queue
        isolated_nodes = set(nx_graph.nodes) - visited
        for isolated_node in isolated_nodes:
            walks.append([isolated_node])
    return walks


# 3. Train Word2Vec (CBOW & Skip-gram)
def train_word2vec(walks, dimensions=200, window_size=5, sg=0, negative=25, epochs=1):
    model = Word2Vec(sentences=walks, vector_size=dimensions, window=window_size, sg=sg, negative=negative,
                     epochs=epochs, workers=4)
    for _ in range(10):
        model.train(walks, total_examples=len(walks), epochs=1)
    return model


# 4. Feature Extraction
def extract_features(model, nodes):
    features, valid_nodes = [], []
    for node in nodes:
        if node in model.wv:
            features.append(model.wv[node])
            valid_nodes.append(node)
    return features, valid_nodes


# Main Execution
graphs = load_rdf_data()
print("RDF Data Loaded!")

SWRC = Namespace("http://swrc.ontoware.org/ontology#")
random.seed(42)
graph = graphs[0]
triples = list(graph)
sampled_triples = random.sample(triples, 2500)
sampled_graph = Graph()
for triple in sampled_triples:
    sampled_graph.add(triple)

kf = KFold(n_splits=10, shuffle=True, random_state=42)

models = [
    ("W2V", "CBOW", 200, 0), ("W2V", "CBOW", 500, 0),
    ("W2V", "SG", 200, 1), ("W2V", "SG", 500, 1),
    ("K2V", "CBOW", 200, 0), ("K2V", "CBOW", 500, 0),
    ("K2V", "SG", 200, 1), ("K2V", "SG", 500, 1)
]

classifiers = {
    "NB": GaussianNB(),
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "SVM": SVC(kernel='linear', C=1.0, random_state=42),
    "C4.5": DecisionTreeClassifier(random_state=42)
}

for model_type, algorithm, dimensions, sg in models:
    print(f"Running model: {model_type} {algorithm} {dimensions}")

    for train_index, test_index in kf.split(sampled_triples):
        train_graph, test_graph = Graph(), Graph()
        for triple in [sampled_triples[i] for i in train_index]:
            train_graph.add(triple)
        for triple in [sampled_triples[i] for i in test_index]:
            test_graph.add(triple)

        bfs_walks = generate_graph_walks([train_graph], depth=8)
        print("BFS Walks Completed!")

        embedding_model = train_word2vec(bfs_walks, dimensions=dimensions, sg=sg)
        print("Embedding Model Trained!")

        affiliations = [(str(s), str(o)) for s, p, o in train_graph if p == SWRC.affiliation]
        X_train_nodes, Y_train_nodes = zip(*affiliations) if affiliations else ([], [])

        filtered_affiliations = [(x, y) for x, y in zip(X_train_nodes, Y_train_nodes) if
                                 x in embedding_model.wv and y in embedding_model.wv]
        filtered_X_train, filtered_Y_train = zip(*filtered_affiliations) if filtered_affiliations else ([], [])

        if not filtered_X_train:
            print("No valid affiliation data in this fold, skipping.")
            continue

        X_train_features, _ = extract_features(embedding_model, filtered_X_train)
        Y_train_features, _ = extract_features(embedding_model, filtered_Y_train)

        affiliations = [(str(s), str(o)) for s, p, o in test_graph if p == SWRC.affiliation]
        X_test_nodes, Y_test_nodes = zip(*affiliations) if affiliations else ([], [])

        filtered_test_affiliations = [(x, y) for x, y in zip(X_test_nodes, Y_test_nodes) if
                                      x in embedding_model.wv and y in embedding_model.wv]
        filtered_X_test, filtered_Y_test = zip(*filtered_test_affiliations) if filtered_test_affiliations else ([], [])

        if not filtered_X_test:
            print("No valid test affiliation data in this fold, skipping.")
            continue

        X_test_features, _ = extract_features(embedding_model, filtered_X_test)
        Y_test_features, _ = extract_features(embedding_model, filtered_Y_test)

        for clf_name, clf in classifiers.items():
            clf.fit(X_train_features, filtered_Y_train)
            predictions = clf.predict(X_test_features)

            accuracy = accuracy_score(filtered_Y_test, predictions)
            print(f"Model {model_type} {algorithm} {dimensions} with {clf_name} Accuracy: {accuracy}")
