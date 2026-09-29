import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

# For Apple
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# ==========================================
# 1. Custom JEV Architecture
# ==========================================
class CustomJev(nn.Module):
    def __init__(self, d_model=768, nhead=8):
        super().__init__()
        
        # 1. State Encoder (Initialized with Pre-Trained DeBERTa)
        # We replace the scratch-built Transformer with DeBERTa-v3-base.
        self.state_encoder = AutoModel.from_pretrained("microsoft/deberta-v3-base")
        
        # 2. Dynamic Cross-Attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, batch_first=True
        )
        
        # 3. Readout Heads
        # Noul Head (Predicts P(True) bounded between 0 and 1)
        self.noul_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        # Choice Projection (Returns raw logits for stable loss calculation)
        self.choice_projection = nn.Linear(d_model, d_model)

    def encode_state(self, input_ids, attention_mask):
        """Passes the JSON string through DeBERTa."""
        outputs = self.state_encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Output shape: [Batch, Seq_Len, d_model]
        return outputs.last_hidden_state

    def evaluate_noul(self, shared_cache, question_embedding):
        """Cross-attends the query against the DeBERTa cache."""
        readout, _ = self.cross_attention(
            query=question_embedding.unsqueeze(1), key=shared_cache, value=shared_cache
        )
        return self.noul_head(readout.squeeze(1))

    def evaluate_choice_logits(self, shared_cache, question_embedding, option_embeddings):
        """Calculates similarities between the state readout and dynamic options."""
        readout, _ = self.cross_attention(
            query=question_embedding.unsqueeze(1), key=shared_cache, value=shared_cache
        )
        projected_readout = self.choice_projection(readout)
        
        # BMM calculates dot-product similarity against all options
        logits = torch.bmm(projected_readout, option_embeddings.transpose(1, 2))
        return logits.squeeze(1)


# ==========================================
# 2. Dataset & Embedding Utils
# ==========================================
class JevDataset(Dataset):
    def __init__(self, json_file):
        with open(json_file, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "state": item["state"],
            "noul_q": item["noul_q"],
            "noul_target": torch.tensor(item["noul_target"], dtype=torch.float32),
            "choice_q": item["choice_q"],
            "options": item["options"],
            "choice_target": torch.tensor(item["choice_target"], dtype=torch.float32)
        }

def mean_pooling(model_output, attention_mask):
    """Nomic-Embed best practice for extracting sentence vectors."""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def create_mock_data(file_path):
    if not os.path.exists(file_path):
        mock_data = [
            {
                "state": '{"ticket_id": 101, "message": "My laptop won\'t turn on."}',
                "noul_q": "Is this a hardware issue?",
                "noul_target": [0.95],
                "choice_q": "Which department should handle this?",
                "options": ["Hardware Repair", "Software Support", "Billing", "Sales"],
                "choice_target": [0.85, 0.10, 0.02, 0.03]
            }
        ] * 16
        with open(file_path, "w") as f:
            json.dump(mock_data, f, indent=2)


# ==========================================
# 3. Main Training Loop
# ==========================================
def train_jev():

    # For Apple Silicon (MPS) or CPU fallback
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # For CUDA
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    data_path = "training_data.json"
    create_mock_data(data_path)
    dataset = JevDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 2. Setup Frozen Nomic Encoder for Queries and Options
    print("Loading nomic-ai/nomic-embed-text-v1.5...")
    tokenizer_q = AutoTokenizer.from_pretrained('nomic-ai/nomic-embed-text-v1.5')
    
    # For CUDA
    # model_q = AutoModel.from_pretrained('nomic-ai/nomic-embed-text-v1.5').to(device)

    # For Apple
    model_q = AutoModel.from_pretrained('nomic-ai/nomic-embed-text-v1.5').to(device, dtype=torch.float32)

    model_q.eval() # Freeze Nomic

    def encode_text(texts, prefix="search_document: "):
        """
        Helper to generate frozen 768d embeddings using Nomic.
        Requires prefixes to activate the correct task space.
        """
        prefixed_texts = [prefix + t for t in texts]
        inputs = tokenizer_q(prefixed_texts, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model_q(**inputs)
        return mean_pooling(outputs, inputs['attention_mask'])

    # 3. Setup Custom JEV Model (Initialized with DeBERTa)
    print("Initializing Custom JEV with microsoft/deberta-v3-base...")
    tokenizer_state = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

    # For CUDA
    # model = CustomJev(d_model=768).to(device)

    # For Apple
    model = CustomJev(d_model=768).to(device, dtype=torch.float32)
    
    optimizer = torch.optim.Adam([
        {'params': model.state_encoder.parameters(), 'lr': 2e-5},
        {'params': model.cross_attention.parameters(), 'lr': 1e-4},
        {'params': model.noul_head.parameters(), 'lr': 1e-4},
        {'params': model.choice_projection.parameters(), 'lr': 1e-4}
    ])

    # 4. Train
    EPOCHS = 5
    model.train()
    
    print("Beginning Training...")
    for epoch in range(EPOCHS):
        total_loss = 0
        
        for batch in dataloader:
            optimizer.zero_grad()
            
            # -- DeBERTa State Prep --
            state_inputs = tokenizer_state(
                batch["state"], padding="max_length", max_length=256, truncation=True, return_tensors="pt"
            ).to(device)
            
            # -- Nomic Query & Option Prep (Using Prefixes) --
            noul_q_embs = encode_text(batch["noul_q"], prefix="search_query: ")
            choice_q_embs = encode_text(batch["choice_q"], prefix="search_query: ")
            
            num_options = len(batch["options"])
            flat_options = [opt for opt_tuple in zip(*batch["options"]) for opt in opt_tuple]
            # Options act as documents we are matching the query against
            flat_option_embs = encode_text(flat_options, prefix="search_document: ")
            option_embs = flat_option_embs.view(len(batch["state"]), num_options, 768)
            
            # Targets
            noul_targets = batch["noul_target"].to(device)
            choice_targets = batch["choice_target"].to(device)

            # -- Forward Pass --
            shared_cache = model.encode_state(
                state_inputs["input_ids"], state_inputs["attention_mask"]
            )
            
            noul_preds = model.evaluate_noul(shared_cache, noul_q_embs)
            loss_noul = F.binary_cross_entropy(noul_preds, noul_targets)
            
            choice_logits = model.evaluate_choice_logits(shared_cache, choice_q_embs, option_embs)
            loss_choice = F.cross_entropy(choice_logits, choice_targets)
            
            # -- Backprop --
            loss = loss_noul + loss_choice
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{EPOCHS} - Average Loss: {total_loss / len(dataloader):.4f}")

    # 5. Save Model
    save_path = "jev_deberta_weights.pt"
    torch.save( model.state_dict(), save_path )
    print(f"Training complete. Weights saved to {save_path}")

if __name__ == "__main__":
    train_jev()