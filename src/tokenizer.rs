// ============================================================================
// tokenizer.rs — BPE Tokenizer (Pure Rust)
// ============================================================================
//
// GGUF models store tokenizer data in metadata:
//   - "tokenizer.ggml.tokens":   string array (vocabulary)
//   - "tokenizer.ggml.scores":   f32 array (token scores, optional)
//   - "tokenizer.ggml.token_type": i32 array (token types)
//   - "tokenizer.ggml.merges":   string array (BPE merge rules, "token_a token_b")
//   - "tokenizer.ggml.pre":      string (pre-tokenizer type)
//
// BPE Tokenization Algorithm:
//   1. Pre-tokenize: split input into initial tokens (byte-level)
//   2. Build initial token set from UTF-8 bytes
//   3. Iteratively apply the highest-priority merge rule
//   4. Map final tokens to vocab IDs
//
// Token types:
//   0 = UNDEFINED, 1 = NORMAL, 2 = UNKNOWN, 3 = CONTROL, 4 = BYTE

use std::collections::HashMap;

use crate::gguf::MetadataValue;

// ============================================================================
// Token Types
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenType {
    Undefined = 0,
    Normal = 1,
    Unknown = 2,
    Control = 3,
    Byte = 4,
}

impl TokenType {
    pub fn from_i32(v: i32) -> Self {
        match v {
            1 => Self::Normal,
            2 => Self::Unknown,
            3 => Self::Control,
            4 => Self::Byte,
            _ => Self::Undefined,
        }
    }
}

// ============================================================================
// Token Entry
// ============================================================================

#[derive(Debug, Clone)]
pub struct TokenEntry {
    pub text: String,
    pub score: f32,
    pub token_type: TokenType,
}

// ============================================================================
// BPE Tokenizer
// ============================================================================

#[derive(Debug, Clone)]
pub struct Tokenizer {
    /// Vocabulary: token_id -> token text
    pub vocab: Vec<TokenEntry>,
    /// Reverse lookup: token text -> token_id
    pub vocab_map: HashMap<String, usize>,
    /// BPE merge rules: "token_a token_b" -> priority (lower = higher priority)
    pub merges: HashMap<(String, String), usize>,
    /// Pre-tokenizer type
    pub pre_type: String,
    /// Special tokens
    pub bos_token_id: Option<usize>,
    pub eos_token_id: Option<usize>,
}

impl Tokenizer {
    /// Build tokenizer from GGUF metadata
    pub fn from_gguf_metadata(metadata: &HashMap<String, MetadataValue>) -> Self {
        // --- Vocabulary ---
        let tokens = metadata
            .get("tokenizer.ggml.tokens")
            .and_then(|v| v.to_array())
            .expect("Missing tokenizer.ggml.tokens");

        let empty_scores: Vec<MetadataValue> = vec![];
        let scores = metadata
            .get("tokenizer.ggml.scores")
            .and_then(|v| v.to_array())
            .unwrap_or(&empty_scores);

        let empty_types: Vec<MetadataValue> = vec![];
        let token_types = metadata
            .get("tokenizer.ggml.token_type")
            .and_then(|v| v.to_array())
            .unwrap_or(&empty_types);

        let mut vocab = Vec::with_capacity(tokens.len());
        let mut vocab_map = HashMap::new();

        for (i, tok_val) in tokens.iter().enumerate() {
            let text = tok_val.to_string_ref().unwrap_or("").to_string();
            let score = scores.get(i).and_then(|s| s.to_f32()).unwrap_or(0.0);
            let ttype = token_types
                .get(i)
                .and_then(|t| t.to_i32())
                .unwrap_or(0);

            vocab_map.insert(text.clone(), i);
            vocab.push(TokenEntry {
                text,
                score,
                token_type: TokenType::from_i32(ttype),
            });
        }

        // --- BPE Merges ---
        let empty_merges: Vec<MetadataValue> = vec![];
        let merges_arr = metadata
            .get("tokenizer.ggml.merges")
            .and_then(|v| v.to_array())
            .unwrap_or(&empty_merges);

        let mut merges = HashMap::new();
        for (priority, merge_val) in merges_arr.iter().enumerate() {
            if let Some(merge_str) = merge_val.to_string_ref() {
                let parts: Vec<&str> = merge_str.splitn(2, ' ').collect();
                if parts.len() == 2 {
                    merges.insert((parts[0].to_string(), parts[1].to_string()), priority);
                }
            }
        }

        // --- Pre-tokenizer type ---
        let pre_type = metadata
            .get("tokenizer.ggml.pre")
            .and_then(|v| v.to_string_ref())
            .unwrap_or("unknown")
            .to_string();

        // --- Special tokens ---
        let bos_token_id = metadata
            .get("tokenizer.ggml.bos_token_id")
            .or_else(|| metadata.get("tokenizer.ggml.bos_id"))
            .and_then(|v| v.to_i32())
            .map(|v| v as usize);

        let eos_token_id = metadata
            .get("tokenizer.ggml.eos_token_id")
            .or_else(|| metadata.get("tokenizer.ggml.eos_id"))
            .and_then(|v| v.to_i32())
            .map(|v| v as usize);

        eprintln!("Tokenizer loaded:");
        eprintln!("  Vocab size: {}", vocab.len());
        eprintln!("  Merge rules: {}", merges.len());
        eprintln!("  Pre-type: {}", pre_type);
        eprintln!("  BOS token: {:?}", bos_token_id);
        eprintln!("  EOS token: {:?}", eos_token_id);

        Tokenizer {
            vocab,
            vocab_map,
            merges,
            pre_type,
            bos_token_id,
            eos_token_id,
        }
    }

    /// Encode text to token IDs
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let pre_tokens = self.pre_tokenize(text);
        let mut all_ids = Vec::new();

        for piece in &pre_tokens {
            let piece_ids = self.bpe_encode_piece(piece);
            all_ids.extend(piece_ids);
        }

        all_ids
    }

    /// Decode token IDs back to text
    pub fn decode(&self, ids: &[usize]) -> String {
        let mut result = String::new();
        for &id in ids {
            if id < self.vocab.len() {
                let text = &self.vocab[id].text;
                // Skip control tokens (like BOS/EOS)
                if self.vocab[id].token_type != TokenType::Control {
                    result.push_str(text);
                }
            }
        }
        result
    }

    /// Pre-tokenize: split input into pieces based on the pre-tokenizer type
    fn pre_tokenize(&self, text: &str) -> Vec<String> {
        match self.pre_type.as_str() {
            "llama-bpe" | "refact" | "tekken" | "qwen2" => self.pre_tokenize_bpe(text),
            "default" | "" => self.pre_tokenize_default(text),
            _ => {
                // Fallback: use BPE pre-tokenization for most models
                self.pre_tokenize_bpe(text)
            }
        }
    }

    /// Default pre-tokenizer: split on whitespace
    fn pre_tokenize_default(&self, text: &str) -> Vec<String> {
        text.split_whitespace()
            .map(|s| s.to_string())
            .filter(|s| !s.is_empty())
            .collect()
    }

    /// BPE-style pre-tokenizer: split into word pieces, respecting punctuation
    /// This implements a simplified LLaMA-style pre-tokenization
    fn pre_tokenize_bpe(&self, text: &str) -> Vec<String> {
        let mut pieces = Vec::new();
        let mut current = String::new();

        for ch in text.chars() {
            if ch.is_ascii_alphanumeric() {
                current.push(ch);
            } else if ch.is_ascii_whitespace() {
                if !current.is_empty() {
                    pieces.push(std::mem::take(&mut current));
                }
                pieces.push(ch.to_string());
            } else if ch.is_ascii_punctuation() {
                if !current.is_empty() {
                    pieces.push(std::mem::take(&mut current));
                }
                pieces.push(ch.to_string());
            } else {
                // Non-ASCII: treat as a separate piece
                if !current.is_empty() {
                    pieces.push(std::mem::take(&mut current));
                }
                pieces.push(ch.to_string());
            }
        }
        if !current.is_empty() {
            pieces.push(current);
        }

        pieces
    }

    /// BPE encode a single pre-tokenized piece
    fn bpe_encode_piece(&self, piece: &str) -> Vec<usize> {
        if piece.is_empty() {
            return vec![];
        }

        // Start with individual bytes
        let bytes: Vec<u8> = piece.bytes().collect();

        // Convert bytes to token strings
        // Check if each byte maps directly to a vocabulary token
        let mut tokens: Vec<String> = Vec::new();
        for &b in &bytes {
            tokens.push(format!("<0x{:02X}>", b));
        }

        // Apply BPE merges
        loop {
            if tokens.len() < 2 {
                break;
            }

            // Find the merge with highest priority (lowest number)
            let mut best_merge: Option<(usize, usize, usize)> = None; // (pos, priority, ...)
            let mut best_priority = usize::MAX;

            for i in 0..tokens.len() - 1 {
                let key = (tokens[i].clone(), tokens[i + 1].clone());
                if let Some(&priority) = self.merges.get(&key) {
                    if priority < best_priority {
                        best_priority = priority;
                        best_merge = Some((i, priority, 0));
                    }
                }
            }

            match best_merge {
                Some((pos, _, _)) => {
                    // Merge the pair at `pos`
                    let merged = format!("{}{}", tokens[pos], tokens[pos + 1]);
                    tokens[pos] = merged;
                    tokens.remove(pos + 1);
                }
                None => break, // No more merges possible
            }
        }

        // Convert final tokens to IDs
        tokens
            .iter()
            .map(|t| self.token_to_id(t))
            .collect()
    }

    /// Look up token ID from token text
    fn token_to_id(&self, token: &str) -> usize {
        if let Some(&id) = self.vocab_map.get(token) {
            return id;
        }
        // Try the token text directly (not byte-escaped)
        if let Some(&id) = self.vocab_map.get(token) {
            return id;
        }
        // Fallback: return 0 (UNK) or the token as-is
        if let Some(&id) = self.vocab_map.get("<unk>") {
            return id;
        }
        0 // Unknown token
    }

    /// Get token text by ID
    pub fn id_to_token(&self, id: usize) -> Option<&str> {
        self.vocab.get(id).map(|t| t.text.as_str())
    }

    /// Vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}
