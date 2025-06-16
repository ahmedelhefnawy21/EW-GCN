import spacy
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.datasets import fetch_20newsgroups
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split


# Load the spaCy model with medium-sized English vectors
nlp = spacy.load("en_core_web_md")

def extract_nodes(text):
    """
    Extracts unique entities and content words from a text.

    Args:
        text (str): Input text to process.

    Returns:
        tuple: (nodes, entities, words)
            - nodes: List of unique entities followed by unique words.
            - entities: List of unique named entities.
            - words: List of alphabetic, non-stop words not part of entities.
    """
    doc = nlp(text)
    # 1) Collect unique entities from the text
    entities = [ent.text for ent in doc.ents]
    # 2) Identify spans of entities to exclude their tokens from word list
    ent_spans = {(ent.start, ent.end) for ent in doc.ents}
    words = []
    for token in doc:
        # Include only alphabetic, non-stop words not within entity spans
        if token.is_alpha and not token.is_stop:
            # Check if token is outside all entity spans
            if not any(start <= token.i < end for start, end in ent_spans):
                words.append(token.text.lower())
    # Combine entities and words, removing duplicates while preserving order
    nodes = list(dict.fromkeys(entities + words))
    return nodes, entities, words


# Get the vector length of spaCy embeddings (typically 300 for 'en_core_web_md')
vec_len = nlp.vocab.vectors_length  # 300 (or 50 if using a smaller model)

def get_embedding(token):
    """
    Retrieves the embedding vector for a given token.

    Args:
        token (str): The token to embed.

    Returns:
        Tensor: Embedding vector as a PyTorch tensor.
    """
    return torch.tensor(nlp(token).vector, dtype=torch.float)

def build_graph(text, label, max_words=80, window=2):
    """
    Constructs a graph from text with entities and words as nodes.

    Args:
        text (str): Input text.
        label (int): Class label for the text.
        max_words (int): Maximum number of words to include.
        window (int): Window size for word-word edges.

    Returns:
        Data or None: PyTorch Geometric Data object or None if invalid.
    """
    nodes, entities, words = extract_nodes(text)
    if len(entities) == 0:  # Skip documents with no entities
        return None
    words = words[:max_words]  # Limit the number of words
    if not nodes:  # Skip if no nodes are extracted
        return None

    # Create node features using half-precision embeddings
    x = torch.stack([torch.tensor(nlp(tok).vector, dtype=torch.float16) for tok in nodes])
    edge = []

    # Add bidirectional edges between entities and words
    for e in entities:
        ei = nodes.index(e)
        for w in words:
            wi = nodes.index(w)
            edge += [(ei, wi), (wi, ei)]

    # Add edges between words within a sliding window
    for i in range(len(words) - window + 1):
        span = words[i:i + window]
        for u in span:
            for v in span:
                if u != v:
                    edge.append((nodes.index(u), nodes.index(v)))

    # Convert edges to PyTorch Geometric format
    edge_index = torch.tensor(edge, dtype=torch.long).t().contiguous()
    data = Data(x=x, edge_index=edge_index, y=torch.tensor([label]), num_entity=len(entities))
    return data


# Fetch the 20-Newsgroups dataset, removing headers, footers, and quotes
dataset = fetch_20newsgroups(subset="all", remove=('headers', 'footers', 'quotes'))
texts = dataset.data  # List of document texts
labels = dataset.target  # Integer labels (0–19)

num_labels = len(dataset.target_names)  # Number of unique classes (20)


# Split dataset into training and validation sets (80-20 split)
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# Build graph representations for training data
train_graphs = []
for t, l in zip(train_texts, train_labels):
    data = build_graph(t, l)
    if data is not None:  # Exclude invalid graphs
        train_graphs.append(data)

# Build graph representations for validation data
val_graphs = []
for t, l in zip(val_texts, val_labels):
    data = build_graph(t, l)
    if data is not None:  # Exclude invalid graphs
        val_graphs.append(data)

# Create data loaders for batching
train_loader = DataLoader(train_graphs, batch_size=4, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=4)

from torch_geometric.nn import global_add_pool

class EWGCN(torch.nn.Module):
    """
    Entity-Word Graph Convolutional Network for text classification.
    """
    def __init__(self, in_dim, hidden=64, n_cls=20):
        """
        Initializes the EW-GCN model.

        Args:
            in_dim (int): Input dimension (embedding size).
            hidden (int): Hidden layer size.
            n_cls (int): Number of classes.
        """
        super().__init__()
        self.c1 = GCNConv(in_dim, hidden)  # First GCN layer
        self.c2 = GCNConv(hidden, hidden)  # Second GCN layer
        self.lin = torch.nn.Linear(hidden, n_cls)  # Classification layer

    def forward(self, x, edge_index, batch, num_entity):
        """
        Forward pass of the EW-GCN model.

        Args:
            x (Tensor): Node features.
            edge_index (Tensor): Edge indices.
            batch (Tensor): Batch vector.
            num_entity (Tensor): Number of entities per graph.

        Returns:
            Tensor: Log-probabilities for each class.
        """
        # Apply GCN layers with ReLU activation
        x = self.c1(x.float(), edge_index).relu()
        x = self.c2(x, edge_index).relu()

        # Create a mask to identify entity nodes
        ent_mask = (torch.arange(x.size(0), device=x.device) < num_entity[batch]).float().unsqueeze(-1)

        # Compute sum and count of entity features per graph
        ent_sum = global_add_pool(x * ent_mask, batch)
        ent_cnt = global_add_pool(ent_mask, batch)

        # Compute mean of entity features with a fallback to all nodes
        ent_mean = ent_sum / (ent_cnt + 1e-6)
        all_mean = global_mean_pool(x, batch)
        cond = (ent_cnt > 0).expand_as(ent_mean)
        pooled = torch.where(cond, ent_mean, all_mean)

        # Output class probabilities
        return F.log_softmax(self.lin(pooled), dim=1)

# Set device to GPU (MPS) if available, otherwise CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = EWGCN(vec_len, hidden=64, n_cls=num_labels).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

def train_epoch():
    """
    Trains the model for one epoch.

    Returns:
        float: Average training loss.
    """
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        opt.zero_grad()
        out = model(batch.x.to(device), batch.edge_index.to(device),
                    batch.batch.to(device), batch.num_entity.to(device))
        loss = F.nll_loss(out, batch.y)
        loss.backward()
        opt.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def eval_loader(loader):
    """
    Evaluates the model on a data loader.

    Args:
        loader (DataLoader): Data loader to evaluate.

    Returns:
        float: Accuracy on the dataset.
    """
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index, batch.batch,
                         batch.num_entity.to(device)).argmax(dim=1)
            correct += int((pred == batch.y).sum())
    return correct / len(loader.dataset)

# Train and evaluate the model for 50 epochs
for epoch in range(1, 51):
    loss = train_epoch()
    val_acc = eval_loader(val_loader)
    print(f"Epoch {epoch:02d}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}")
