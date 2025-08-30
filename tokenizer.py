import json
import re

class CustomTokenizer:
    def __init__(self, vocab_path: str):
        # Load vocabulary
        if vocab_path.endswith('.json'):
            with open(vocab_path, "r") as f:
                vocab_data = json.load(f)
                # Handle both formats: direct token_to_id or nested structure
                if 'token_to_id' in vocab_data:
                    self.token_to_id = vocab_data['token_to_id']
                else:
                    self.token_to_id = vocab_data
        else:
            with open(vocab_path, "r") as f:
                self.token_to_id = json.load(f)

        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self.vocab = set(self.token_to_id.keys())

        # Required special tokens
        self.pad_token = "[PAD]"
        self.mask_token = "[MASK]"
        self.unk_token = "[UNK]"
        self.eos_token = "[EOS]"
        self.noagent_token = "[NOAGENT]"

        # IDs for special tokens
        self.pad_token_id = self._require_token(self.pad_token)
        self.mask_token_id = self._require_token(self.mask_token)
        self.unk_token_id = self._require_token(self.unk_token)
        self.eos_token_id = self._require_token(self.eos_token)
        self.noagent_token_id = self._require_token(self.noagent_token)

        # Regex for reaction classes
        self.reaction_class_pattern = re.compile(r'^\d+(\.\d+)*$')
        
        # SMILES tokenization pattern (from your original tokenizer)
        self.SMILES_REGEX = r"(>>|\[[^\[\]]{1,10}\]|Br|Cl|Si|Se|Na|Ca|Li|Mg|Zn|Cu|Fe|Mn|Hg|Ag|Au|[B-IK-Zb-ik-z0-9=#$%@+\->\(\)/\\\.])"
        self.smiles_pattern = re.compile(self.SMILES_REGEX)

    def _require_token(self, token):
        tid = self.token_to_id.get(token, None)
        if tid is None:
            raise ValueError(f"{token} token is missing in vocabulary!")
        return tid

    @property
    def vocab_size(self):
        """Number of tokens in vocab (includes [MASK])"""
        return len(self.token_to_id)

    def get_special_token_id(self, token):
        return self.token_to_id.get(token, self.unk_token_id)

    def is_reaction_class(self, text):
        """Check if text is a reaction class like 2.12.13"""
        return bool(self.reaction_class_pattern.match(text))

    def tokenize_smiles(self, smiles_string):
        """Tokenize SMILES string using regex"""
        return self.smiles_pattern.findall(smiles_string)

    def encode(self, text: str) -> list:
        """
        Enhanced tokenization that handles both SMILES and reaction classes.
        Expects format: "SMILES_STRING" or "SMILES_STRING REACTION_CLASS"
        """
        pieces = text.strip().split()
        tokens = []
        
        for piece in pieces:
            if self.is_reaction_class(piece):
                # Treat reaction class as single token
                tokens.append(piece)
            elif piece in self.vocab:
                # If the whole piece is in vocab (like special tokens)
                tokens.append(piece)
            else:
                # Tokenize as SMILES
                smiles_tokens = self.tokenize_smiles(piece)
                tokens.extend(smiles_tokens)
        
        # Convert tokens to IDs
        return [self.token_to_id.get(token, self.unk_token_id) for token in tokens]

    def decode(self, token_ids: list) -> str:
        """
        Decode token IDs back to string.
        For reaction classes and special tokens, add spaces.
        For SMILES tokens, concatenate without spaces.
        """
        if not token_ids:
            return ""
            
        tokens = []
        for i in token_ids:
            if i == self.pad_token_id:
                continue
            token = self.id_to_token.get(i, self.unk_token)
            tokens.append(token)
        
        if not tokens:
            return ""
        
        # Smart joining: add spaces around reaction classes and special tokens
        result = []
        for i, token in enumerate(tokens):
            if i == 0:
                result.append(token)
            elif (self.is_reaction_class(token) or 
                  self.is_reaction_class(tokens[i-1]) or
                  token.startswith('[') or 
                  tokens[i-1].startswith('[')):
                result.append(' ' + token)
            else:
                result.append(token)
        
        return ''.join(result)

    def _greedy_tokenize(self, text: str) -> list:
        """
        Legacy greedy tokenization - kept for compatibility but not recommended
        for reaction class data. Use encode() instead.
        """
        i = 0
        tokens = []
        while i < len(text):
            match = None
            for j in range(len(text), i, -1):
                sub = text[i:j]
                if sub in self.vocab:
                    match = sub
                    break
            if match:
                tokens.append(match)
                i += len(match)
            else:
                tokens.append(self.unk_token)
                i += 1
        return tokens


# Example usage and testing
if __name__ == "__main__":
    
    # Example test cases
    test_cases = [
        "CCO>NaOH>CCCBr 2.12.13",
        "CCO>>CCCBr 1.5.2",
        "C[N+](C)(C)C.Cl 3.1.1"
    ]

    tokenizer = CustomTokenizer("/home/gpwuq/ipms-foundation_model/data/interim/clean_vocab_reactions_and_classes_combined.json")