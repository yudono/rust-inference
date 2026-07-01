// Chat Template Engine
use crate::gguf::MetadataValue;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone)]
pub struct ChatTemplate {
    pub template_str: String,
    pub messages: Vec<ChatMessage>,
}

impl ChatTemplate {
    pub fn from_metadata(metadata: &HashMap<String, MetadataValue>) -> Option<Self> {
        let template_str = metadata
            .get("tokenizer.chat_template")
            .and_then(|v| v.to_string_ref())?
            .to_string();
        Some(ChatTemplate { template_str, messages: Vec::new() })
    }

    pub fn add_message(&mut self, role: &str, content: &str) {
        self.messages.push(ChatMessage { role: role.to_string(), content: content.to_string() });
    }

    pub fn apply(&self, user_prompt: &str, system_prompt: Option<&str>) -> String {
        self.apply_template(user_prompt, system_prompt)
    }

    fn apply_template(&self, user_prompt: &str, system_prompt: Option<&str>) -> String {
        apply_jinja2_template(&self.template_str, user_prompt, system_prompt)
    }
}

fn detect_template_type(template: &str) -> &'static str {
    let t = template.trim();
    if t.contains("<|im_start|>") || t.contains("<|im_end|>") {
        "chatml"
    } else if t.contains("<|start_header_id|>") || t.contains("<|eot_id|>") {
        "llama3"
    } else if t.contains("[INST]") || t.contains("[/INST]") {
        "llama2"
    } else {
        "chatml"
    }
}

fn apply_jinja2_template(template: &str, user_prompt: &str, system_prompt: Option<&str>) -> String {
    match detect_template_type(template) {
        "chatml" => apply_chatml_fallback(user_prompt, system_prompt),
        "llama3" => apply_llama3_fallback(user_prompt, system_prompt),
        "llama2" => apply_llama2_fallback(user_prompt, system_prompt),
        _ => apply_chatml_fallback(user_prompt, system_prompt),
    }
}

fn apply_chatml_fallback(user_prompt: &str, system_prompt: Option<&str>) -> String {
    let mut result = String::new();
    if let Some(sys) = system_prompt {
        result.push_str("<|im_start|>system\n");
        result.push_str(sys);
        result.push_str("<|im_end|>\n");
    }
    result.push_str("<|im_start|>user\n");
    result.push_str(user_prompt);
    result.push_str("<|im_end|>\n");
    result.push_str("<|im_start|>assistant\n");
    result
}

fn apply_llama3_fallback(user_prompt: &str, system_prompt: Option<&str>) -> String {
    let mut result = String::new();
    result.push_str("<|begin_of_text|>");
    if let Some(sys) = system_prompt {
        result.push_str("<|start_header_id|>system<|end_header_id|>\n\n");
        result.push_str(sys);
        result.push_str("<|eot_id|>");
    }
    result.push_str("<|start_header_id|>user<|end_header_id|>\n\n");
    result.push_str(user_prompt);
    result.push_str("<|eot_id|>");
    result.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    result
}

fn apply_llama2_fallback(user_prompt: &str, system_prompt: Option<&str>) -> String {
    let mut result = String::new();
    result.push_str("[INST] ");
    if let Some(sys) = system_prompt {
        result.push_str("<<SYS>>\n");
        result.push_str(sys);
        result.push_str("\n<</SYS>>\n\n");
    }
    result.push_str(user_prompt);
    result.push_str(" [/INST]");
    result
}
