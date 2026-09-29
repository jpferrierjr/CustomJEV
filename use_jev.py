import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from main import CustomJev

# 1. Force Apple Silicon MPS Backend Fallback
# This prevents crashes when Metal doesn't support specific PyTorch operations natively.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# ==========================================
# Inference Engine Helper
# ==========================================
class JevInferenceEngine:
    def __init__(self, weights_path="jev_deberta_weights.pt"):
        # Auto-detect Apple Silicon, fallback to CPU
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Loading JEV onto {self.device}...")

        # 1. Setup Nomic for Queries & Options (Frozen, FP32)
        print("Loading Nomic Embed for queries...")
        self.tokenizer_q = AutoTokenizer.from_pretrained('nomic-ai/nomic-embed-text-v1.5')
        self.model_q = AutoModel.from_pretrained(
            'nomic-ai/nomic-embed-text-v1.5',
            torch_dtype=torch.float32
        ).to(self.device)
        self.model_q.eval()

        # 2. Setup DeBERTa Tokenizer for State
        self.tokenizer_state = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

        # 3. Setup Custom JEV & Load Weights (FP32)
        print(f"Initializing JEV weights from {weights_path}...")
        self.model = CustomJev(d_model=768).to(self.device, dtype=torch.float32)
        
        if os.path.exists(weights_path):
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
            print("Weights loaded successfully.")
        else:
            print(f"WARNING: {weights_path} not found. Running untrained base model.")
            
        self.model.eval()

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def _encode_text(self, texts, prefix="search_document: "):
        prefixed_texts = [prefix + t for t in texts]
        inputs = self.tokenizer_q(prefixed_texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model_q(**inputs)
        return self._mean_pooling(outputs, inputs['attention_mask'])

    def run_evaluations(self, state_text, noul_question, choice_question, choice_options):
        """Encodes state ONCE, then runs multiple queries against it."""
        print("\n--- Running Inference ---")
        
        with torch.no_grad():
            # 1. Parse Unstructured State
            state_inputs = self.tokenizer_state(
                state_text, padding="max_length", max_length=256, truncation=True, return_tensors="pt"
            ).to(self.device)
            
            # 2. Encode State (Shared Cache)
            shared_cache = self.model.encode_state(
                state_inputs["input_ids"], state_inputs["attention_mask"]
            )

            # 3. Noul (True/False) Evaluation
            noul_q_emb = self._encode_text([noul_question], prefix="search_query: ")
            noul_pred = self.model.evaluate_noul(shared_cache, noul_q_emb)
            prob_true = noul_pred.item()

            # 4. Choice (Categorical) Evaluation
            choice_q_emb = self._encode_text([choice_question], prefix="search_query: ")
            option_embs = self._encode_text(choice_options, prefix="search_document: ")
            # Expand dimensions to match batch size = 1
            option_embs = option_embs.unsqueeze(0) 
            
            choice_logits = self.model.evaluate_choice_logits(shared_cache, choice_q_emb, option_embs)
            choice_probs = F.softmax(choice_logits, dim=-1).squeeze(0).cpu().numpy()

        # Format Results
        results = {
            "noul": {
                "question": noul_question,
                "probability_true": round(prob_true, 4)
            },
            "choice": {
                "question": choice_question,
                "results": {opt: round(float(prob), 4) for opt, prob in zip(choice_options, choice_probs)}
            }
        }
        return results


def ensure_examples_file(file_path):
    """Creates the 10 example JSON test records if the file doesn't exist."""
    if os.path.exists(file_path):
        return

    examples = [
        {
            "title": "E-Commerce (Returns & Triage)",
            "state": '{"order_id": 9923, "customer_message": "My package arrived but the glass vase was completely shattered.", "days_since_delivery": 2, "loyalty_tier": "Gold"}',
            "noul_q": "Is the customer reporting physical damage to the product?",
            "choice_q": "What is the appropriate customer service resolution?",
            "options": ["Issue Full Refund", "Send Replacement", "Deny Request", "Escalate to Shipping Carrier"]
        },
        {
            "title": "Cybersecurity (Threat Hunting)",
            "state": '{"alert_id": "SEC-092", "source_ip": "192.168.1.45", "event": "Multiple failed login attempts followed by successful access and immediate massive data egress.", "user_role": "Guest"}',
            "noul_q": "Does this alert indicate a potential data exfiltration event?",
            "choice_q": "What is the required automated security response?",
            "options": ["Isolate Host", "Block IP Address", "Reset User Password", "Mark as False Positive"]
        },
        {
            "title": "Medical / Healthcare (Patient Triage)",
            "state": '{"patient_age": 54, "symptoms": ["chest tightness", "shortness of breath", "nausea"], "history": ["hypertension"], "vitals": {"bp": "160/95", "hr": 110}}',
            "noul_q": "Are the symptoms indicative of a potential cardiac event?",
            "choice_q": "What triage priority level should be assigned to this patient?",
            "options": ["Level 1 - Resuscitation", "Level 2 - Emergent", "Level 3 - Urgent", "Level 4 - Non-Urgent"]
        },
        {
            "title": "DevOps (Incident Routing)",
            "state": '{"service": "auth-gateway", "error_code": "502 Bad Gateway", "logs": "Connection pool exhausted. Unable to reach database backend.", "environment": "Production"}',
            "noul_q": "Is this incident causing a complete service outage for users?",
            "choice_q": "Which engineering team should be paged?",
            "options": ["Database Reliability", "Frontend", "Network Security", "Cloud Infrastructure"]
        },
        {
            "title": "Fraud Detection (Fintech)",
            "state": '{"transaction_id": "TX-9912", "amount": 4500.00, "merchant": "Luxury Watches Intl", "user_location": "US", "ip_location": "RU", "velocity": "3 transactions in 10 minutes"}',
            "noul_q": "Is there a geographical mismatch between the user's registered location and the IP address?",
            "choice_q": "What automated fraud prevention action should be taken?",
            "options": ["Approve Transaction", "Require SMS 2FA", "Decline and Lock Card", "Flag for Manual Review"]
        },
        {
            "title": "Logistics & Supply Chain",
            "state": '{"shipment_id": "TRK-882", "current_location": "Denver, CO", "weather": "Severe Blizzard", "contents": "Perishable pharmaceuticals", "temp_sensor": "Normal"}',
            "noul_q": "Is the shipment currently experiencing a temperature control failure?",
            "choice_q": "What routing adjustment should be applied?",
            "options": ["Reroute to Climate-Controlled Warehouse", "Expedite via Air Freight", "Maintain Current Route", "Return to Sender"]
        },
        {
            "title": "Human Resources (Applicant Tracking)",
            "state": '{"candidate": "APP-229", "role": "Senior Backend Engineer", "experience": {"python": 5, "go": 1, "kubernetes": 3}, "education": "B.S. Computer Science"}',
            "noul_q": "Does the candidate meet a minimum requirement of 4 years of Python experience?",
            "choice_q": "What is the recommended next step for this application?",
            "options": ["Schedule Technical Screen", "Send HackerRank Assessment", "Reject due to under-qualification", "Keep on file for future"]
        },
        {
            "title": "Legal & Compliance",
            "state": '{"document_type": "Vendor MSA", "clause_4": "Vendor reserves the right to use anonymized customer data for internal AI model training.", "jurisdiction": "EU (GDPR)"}',
            "noul_q": "Does this clause potentially violate GDPR restrictions on data processing?",
            "choice_q": "Which legal department needs to review this document?",
            "options": ["Data Privacy & Compliance", "Intellectual Property", "Labor & Employment", "Standard Procurement"]
        },
        {
            "title": "IoT & Smart Home Devices",
            "state": '{"device_type": "Smart Thermostat", "firmware": "v2.1.4", "sensor_reading": "Error -99", "ambient_temp_reported": null, "wifi_strength": "Excellent"}',
            "noul_q": "Is the device experiencing a network connectivity issue?",
            "choice_q": "What command should the central hub issue to the device?",
            "options": ["Reboot Device", "Factory Reset", "Push Firmware Update", "Enter Failsafe Mode"]
        },
        {
            "title": "SaaS Customer Success (Churn Prediction)",
            "state": '{"account_id": "ACME-Corp", "mrr": "$2000", "active_users": 2, "total_seats": 50, "last_login": "14 days ago", "support_tickets_open": 3}',
            "noul_q": "Is this account exhibiting signs of high churn risk?",
            "choice_q": "What intervention strategy should be applied?",
            "options": ["Automated Re-engagement Email", "Assign Dedicated Success Manager", "Offer Discounted Renewal", "No Action Needed"]
        }
    ]

    print(f"Creating examples file at {file_path}...")
    with open(file_path, "w") as f:
        json.dump(examples, f, indent=4)


# ==========================================
# Run Example Prompt
# ==========================================
if __name__ == "__main__":
    file_name = "test_examples.json"
    ensure_examples_file(file_name)

    engine = JevInferenceEngine()
    
    with open(file_name, "r") as f:
        tests = json.load(f)

    for i, test in enumerate(tests, 1):
        print(f"\n==========================================")
        print(f" Test {i}: {test['title']}")
        print(f"==========================================")
        print(f"STATE PAYLOAD:\n{test['state']}\n")
        
        outputs = engine.run_evaluations(
            state_text=test['state'],
            noul_question=test['noul_q'],
            choice_question=test['choice_q'],
            choice_options=test['options']
        )
        
        print(f"[NOUL] {outputs['noul']['question']}")
        print(f"-> P(True): {outputs['noul']['probability_true'] * 100:.1f}%\n")
        
        print(f"[CHOICE] {outputs['choice']['question']}")
        
        # Sort results by highest probability for display
        sorted_choices = sorted(outputs['choice']['results'].items(), key=lambda x: x[1], reverse=True)
        for option, prob in sorted_choices:
            print(f"-> {option}: {prob * 100:.1f}%")