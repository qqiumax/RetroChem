import subprocess

'''
# Install required packages
def install_packages():
    packages = [
        'rdkit',
        'torch',
        'pandas',
        'numpy',
        'tqdm',
        'urllib3'
    ]
    for pkg in packages:
        try:
            subprocess.check_call(['pip', 'install', pkg])
        except subprocess.CalledProcessError:
            print(f"Failed to install {pkg}")

install_packages()
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np
import os
import urllib.request
from datasets import load_dataset
from tqdm import tqdm


class RetrosynthesisTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=d_model*4
        )
        self.fc_out = nn.Linear(d_model, vocab_size)
        
    def forward(self, src, tgt):
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)
        output = self.transformer(src, tgt)
        return self.fc_out(output)

class ReactionDataset(Dataset):
    def __init__(self, reactions, char_to_idx, max_len=200):
        self.reactions = reactions
        self.char_to_idx = char_to_idx
        self.idx_to_char = {v: k for k, v in char_to_idx.items()}
        self.max_len = max_len
        
    def __len__(self):
        return len(self.reactions)
    
    def __getitem__(self, idx):
        product, reactants = self.reactions[idx]
        product_seq = self.smi_to_seq(product)
        reactant_seq = self.smi_to_seq(reactants)
        return torch.LongTensor(product_seq), torch.LongTensor(reactant_seq)
    
    def smi_to_seq(self, smi):
        seq = [self.char_to_idx[c] for c in smi]
        seq = seq[:self.max_len-1]
        seq += [self.char_to_idx['<EOS>']]
        if len(seq) < self.max_len:
            seq += [self.char_to_idx['<PAD>']] * (self.max_len - len(seq))
        return seq[:self.max_len]

def prepare_data():
    # Load dataset from Hugging Face Hub in Parquet format
    dataset = load_dataset("Phando/uspto-50k")
    
    reactions = []
    chars = {'<PAD>', '<SOS>', '<EOS>'}  # Initialize with special tokens
    
    for split in ['train', 'validation']:
        for item in dataset[split]:
            # Extract product and reactants from dedicated columns
            product = item['prod_smiles'].strip()
            reactants = item['rxn_smiles'].strip()
            
            # Add to reactions list (product -> reactants mapping)
            reactions.append((product, reactants))
            
            # Update character vocabulary
            chars.update(product)
            chars.update(reactants)
    
    # Create comprehensive character mapping
    all_chars = sorted(chars)
    char_to_idx = {char: idx for idx, char in enumerate(all_chars)}
    
    # Verify special tokens positions
    assert '<PAD>' in char_to_idx, "Special token missing in vocabulary"
    assert '<SOS>' in char_to_idx, "Special token missing in vocabulary"
    assert '<EOS>' in char_to_idx, "Special token missing in vocabulary"
    
    return reactions, char_to_idx

def train_model():
    print(1)
    reactions, char_to_idx = prepare_data()
    vocab_size = len(char_to_idx)
    
    dataset = ReactionDataset(reactions, char_to_idx)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RetrosynthesisTransformer(vocab_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss(ignore_index=char_to_idx['<PAD>'])
    
    for epoch in range(10):
        model.train()
        total_loss = 0
        for src, tgt in tqdm(dataloader):
            src = src.to(device)
            tgt = tgt.to(device)
            
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:].contiguous().view(-1)
            
            output = model(src, tgt_input)
            output = output.view(-1, vocab_size)
            
            loss = criterion(output, tgt_output)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}")
    
    torch.save(model.state_dict(), 'retrosynthesis_model.pth')
    return model, char_to_idx

def predict_reactants(mol_file, model, char_to_idx, max_depth=3):
    mol = Chem.MolFromMolFile(mol_file)
    if mol is None:
        raise ValueError("Invalid .mol file")
    target_smi = Chem.MolToSmiles(mol)
    
    idx_to_char = {v: k for k, v in char_to_idx.items()}
    device = next(model.parameters()).device
    
    def generate_step(smi):
        seq = [char_to_idx[c] for c in smi if c in char_to_idx]
        seq = [char_to_idx['<SOS>']] + seq + [char_to_idx['<EOS>']]
        seq = torch.LongTensor(seq).unsqueeze(0).to(device)
        
        memory = model.transformer.encoder(model.embedding(seq.permute(1, 0)))
        outputs = []
        next_token = torch.LongTensor([[char_to_idx['<SOS>']]]).to(device)
        
        for _ in range(100):
            output = model.transformer.decoder(model.embedding(next_token), memory)
            logits = model.fc_out(output)
            next_token = logits.argmax(-1)[-1:]
            outputs.append(next_token.item())
            if outputs[-1] == char_to_idx['<EOS>']:
                break
        
        return ''.join([idx_to_char[i] for i in outputs if i not in 
                      [char_to_idx['<SOS>'], char_to_idx['<EOS>'], char_to_idx['<PAD>']]])
    
    def recursive_predict(smi, depth=3):
        if depth >= max_depth:
            return {'molecule': smi, 'children': []}
        
        reactants = generate_step(smi)
        children = []
        for reactant in reactants.split('.'):
            children.append(recursive_predict(reactant.strip(), depth+1))
        return {'molecule': smi, 'children': children}
    
    return recursive_predict(target_smi)

def main():
    # Train or load model
    model_path = 'retrosynthesis_model.pth'
    if not os.path.exists(model_path):
        model, char_to_idx = train_model()
    else:
        reactions, char_to_idx = prepare_data()
        vocab_size = len(char_to_idx)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = RetrosynthesisTransformer(vocab_size).to(device)
        model.load_state_dict(torch.load(model_path))
    
    # Example usage
    synthesis_tree = predict_reactants('target.mol', model, char_to_idx)
    
    # Print synthesis tree
    def print_tree(node, indent=0):
        print(' ' * indent + node['molecule'])
        for child in node['children']:
            print_tree(child, indent+4)
    
    print("Predicted Synthesis Route:")
    print_tree(synthesis_tree)

if __name__ == "__main__":
    main()