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

// GPT-2 / tiktoken byte-to-unicode encoder table
fn bytes_to_unicode() -> (Vec<u8>, Vec<char>) {
    // Collect all bytes that are "printable" in Latin-1
    let mut bs: Vec<u8> = (b'!'..=b'~').collect();     // 33..126
    bs.extend(b'\xa1'..=b'\xac');                        // 161..172
    bs.extend(b'\xae'..=b'\xff');                        // 174..255
    // cs starts as a copy of bs (safe bytes map to themselves)
    let mut cs: Vec<char> = bs.iter().map(|&b| std::char::from_u32(b as u32).unwrap()).collect();
    let initial_len = bs.len();
    // Append remaining bytes (0..32, 127..160, 173) with codepoints 256+
    let mut n = 0u32;
    for b in 0..=255u8 {
        if !bs[..initial_len].contains(&b) {
            bs.push(b);
            cs.push(std::char::from_u32(256 + n).unwrap());
            n += 1;
        }
    }
    (bs, cs)
}

fn byte_decoder() -> HashMap<char, u8> {
    let (bs, cs) = bytes_to_unicode();
    cs.iter().zip(bs.iter()).map(|(c, b)| (*c, *b)).collect()
}

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
    /// Token IDs for common stop tokens (e.g., ChatML end-of-turn)
    pub im_end_token_id: Option<usize>,
    pub im_start_token_id: Option<usize>,
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

        // Look up common chat template stop tokens from vocab
        let im_end_token_id = vocab_map.get("<|im_end|>").copied();
        let im_start_token_id = vocab_map.get("<|im_start|>").copied();

        Tokenizer {
            vocab,
            vocab_map,
            merges,
            pre_type,
            bos_token_id,
            eos_token_id,
            im_end_token_id,
            im_start_token_id,
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

    /// Decode token IDs back to text (handles Qwen2 byte-level BPE)
    pub fn decode(&self, ids: &[usize]) -> String {
        let decoder = byte_decoder();
        let mut raw = String::new();
        for &id in ids {
            if id >= self.vocab.len() {
                continue;
            }
            let token = &self.vocab[id];
            match token.token_type {
                TokenType::Control => {}
                _ => raw.push_str(&token.text),
            }
        }
        // Convert byte-encoded chars back to bytes
        let bytes: Vec<u8> = raw.chars().filter_map(|c| decoder.get(&c).copied()).collect();
        String::from_utf8(bytes).unwrap_or_else(|e| {
            let bytes = e.into_bytes();
            String::from_utf8_lossy(&bytes).to_string()
        })
    }
    
    /// Decode a single token ID to raw bytes (for streaming)
    pub fn decode_token_bytes(&self, id: usize) -> Vec<u8> {
        let decoder = byte_decoder();
        if id >= self.vocab.len() {
            return Vec::new();
        }
        let token = &self.vocab[id];
        match token.token_type {
            TokenType::Control => Vec::new(),
            _ => token.text.chars().filter_map(|c| decoder.get(&c).copied()).collect(),
        }
    }
    
    /// Decode a single token ID to text (for streaming)
    pub fn decode_token(&self, id: usize) -> Option<String> {
        Some(self.decode(&[id]))
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
        // GPT-2/tiktoken style: spaces before words are attached to the word.
        // We scan character-by-character, accumulating into a current piece.
        let mut pieces = Vec::new();
        let mut current = String::new();
        let mut pending_space = false;
        let mut chars = text.chars().peekable();

        while let Some(ch) = chars.next() {
            if ch.is_ascii_whitespace() {
                // Whitespace: flush current word and remember for prefix
                if !current.is_empty() {
                    pieces.push(std::mem::take(&mut current));
                }
                // Check if this space is followed by a word/letter
                pending_space = true;
            } else {
                // Non-whitespace character
                if pending_space && (ch.is_ascii_alphanumeric() || !ch.is_ascii()) {
                    // Prepend the space to this word piece
                    current.push(' ');
                    pending_space = false;
                } else if pending_space {
                    // Space before punctuation/other: flush space first
                    pieces.push(" ".to_string());
                    pending_space = false;
                }

                if ch.is_ascii_alphanumeric() {
                    current.push(ch);
                } else if ch.is_ascii_punctuation() {
                    // Punctuation: flush current and push as separate piece
                    if !current.is_empty() {
                        pieces.push(std::mem::take(&mut current));
                    }
                    pieces.push(ch.to_string());
                } else {
                    // Non-ASCII: flush current and push as separate piece
                    if !current.is_empty() {
                        pieces.push(std::mem::take(&mut current));
                    }
                    pieces.push(ch.to_string());
                }
            }
        }

        // Flush remaining
        if !current.is_empty() {
            pieces.push(current);
        }
        if pending_space {
            pieces.push(" ".to_string());
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

        // Convert bytes to token strings for BPE (byte-level / tiktoken style).
        // For bytes 0-255, the corresponding token text is the Unicode character
        // at that codepoint (Latin-1 interpretation for bytes 128-255).
        let mut tokens: Vec<String> = Vec::new();
        for &b in &bytes {
            let raw = char::from_u32(b as u32).unwrap().to_string();
            if self.vocab_map.contains_key(&raw) {
                tokens.push(raw);
            } else {
                // Fallback: use explicit byte encoding as char(256+b)
                let byte_token = char::from_u32(256 + b as u32).unwrap().to_string();
                tokens.push(byte_token);
            }
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
