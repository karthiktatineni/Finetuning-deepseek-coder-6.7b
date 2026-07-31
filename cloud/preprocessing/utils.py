#!/usr/bin/env python3
"""
Utility functions for preprocessing pipeline.
Common functions used across preprocessing scripts.
"""

import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any


def normalize_text(text: str) -> str:
    """Normalize text content for consistency."""
    if text is None:
        return ""
    
    text = str(text).strip()
    
    # Remove excessive newlines (more than 2 consecutive)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove excessive spaces (more than 1 space)
    text = re.sub(r' {2,}', ' ', text)
    
    # Normalize quotes to standard ASCII quotes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    
    # Normalize dashes
    text = text.replace('–', '-').replace('—', '--')
    
    # Remove control characters except newlines, tabs, and carriage returns
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    
    # Remove zero-width spaces
    text = text.replace('\u200b', '').replace('\u200c', '').replace('\u200d', '')
    
    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    return text


def clean_code_snippets(text: str) -> str:
    """Clean code snippets while preserving functionality."""
    if not text:
        return ""
    
    # Preserve common code patterns
    text = text.replace('<tab>', '\t')
    text = text.replace('<br>', '\n')
    text = text.replace('<br/>', '\n')
    
    return text


def extract_code_blocks(text: str) -> List[str]:
    """Extract code blocks from markdown-formatted text."""
    code_blocks = []
    
    # Match code blocks with language specifiers
    pattern = r'```(\w+)?\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    
    for language, code in matches:
        code_blocks.append({
            'language': language.strip() if language else 'unknown',
            'code': code.strip()
        })
    
    return code_blocks


def format_conversation_text(conversations: List[Dict[str, str]], 
                            system_prompt: str = None) -> str:
    """Format conversations into training text format."""
    if not conversations or not isinstance(conversations, list):
        return ""
    
    formatted_parts = []
    
    # Add system prompt if provided
    if system_prompt:
        formatted_parts.append(f"<|System|>: {system_prompt}")
    
    # Format each message
    for message in conversations:
        if not isinstance(message, dict):
            continue
        
        role = message.get('from', message.get('role', ''))
        content = message.get('value', message.get('content', ''))
        
        if not content:
            continue
        
        # Normalize role names
        if role in ['human', 'user', 'instruction']:
            formatted_role = "<|User|>"
        elif role in ['gpt', 'assistant', 'response']:
            formatted_role = "<|Assistant|>"
        elif role in ['system']:
            formatted_role = "<|System|>"
        else:
            formatted_role = f"<|{role.capitalize()}|>"
        
        formatted_parts.append(f"{formatted_role}: {content}")
    
    return '\n'.join(formatted_parts)


def validate_conversation_structure(conversations: List[Dict[str, str]]) -> bool:
    """Validate conversation structure for training."""
    if not conversations or not isinstance(conversations, list):
        return False
    
    if len(conversations) == 0:
        return False
    
    # Check first message is from user
    first_role = conversations[0].get('from', conversations[0].get('role', ''))
    if first_role not in ['human', 'user', 'instruction']:
        return False
    
    # Check each message has required fields
    for msg in conversations:
        if not isinstance(msg, dict):
            return False
        
        has_role = 'from' in msg or 'role' in msg
        has_content = 'value' in msg or 'content' in msg
        
        if not has_role or not has_content:
            return False
        
        content = msg.get('value', msg.get('content', ''))
        if not isinstance(content, str) or not content.strip():
            return False
    
    return True


def count_tokens_estimation(tokenizer, text: str) -> int:
    """Estimate token count without full tokenization."""
    if not text:
        return 0
    
    # Rough estimation: ~4 characters per token for English text
    char_count = len(text)
    estimated_tokens = int(char_count / 4)
    
    return max(estimated_tokens, 1)


def filter_long_examples(examples: List[Dict], max_length: int, 
                        tokenizer=None) -> List[Dict]:
    """Filter out examples that are too long."""
    filtered_examples = []
    
    for example in examples:
        # Determine text length
        if isinstance(example, dict) and 'conversations' in example:
            text = format_conversation_text(example['conversations'])
        elif isinstance(example, str):
            text = example
        else:
            continue
        
        # Check length
        if tokenizer:
            token_count = len(tokenizer.encode(text, truncation=False))
        else:
            token_count = count_tokens_estimation(tokenizer, text)
        
        if token_count <= max_length:
            filtered_examples.append(example)
    
    return filtered_examples


def apply_chat_template(conversations: List[Dict[str, str]], 
                       template_type: str = "default") -> str:
    """Apply different chat templates."""
    templates = {
        "default": format_conversation_text,
        "deepseek": lambda convs: format_conversation_text(convs, system_prompt="You are a helpful assistant."),
        "minimal": lambda convs: "\n\n".join([
            f"{msg.get('from', 'user')}: {msg.get('value', '')}"
            for msg in convs
        ]),
    }
    
    template_func = templates.get(template_type, templates["default"])
    return template_func(conversations)


def remove_personal_info(text: str) -> str:
    """Remove potential personal information from text."""
    # Simple pattern matching for common personal info
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
    
    # Replace with placeholders
    text = re.sub(email_pattern, '[EMAIL]', text)
    text = re.sub(phone_pattern, '[PHONE]', text)
    
    return text


def truncate_conversation(conversations: List[Dict[str, str]], 
                        max_turns: int = 10) -> List[Dict[str, str]]:
    """Limit conversation to maximum number of turns."""
    if len(conversations) <= max_turns:
        return conversations
    
    # Keep the first and last parts
    keep_middle = max_turns - 2  # Save room for first and last messages
    if keep_middle <= 0:
        return [conversations[0], conversations[-1]]
    
    start = conversations[:1]
    middle = conversations[len(conversations)//2 - keep_middle//2 : len(conversations)//2 + keep_middle//2 + 1]
    end = conversations[-1:]
    
    return start + middle + end


def estimate_dataset_memory(dataset: List[Dict], avg_tokens: int = 512) -> float:
    """Estimate memory requirements for dataset storage."""
    num_examples = len(dataset)
    
    # Rough estimates
    tokens_per_example = avg_tokens
    examples_per_million_tokens = 1000000 / tokens_per_example
    
    # Memory estimates (assuming 4 bytes per token)
    memory_gb = (num_examples * tokens_per_example * 4) / (1024 ** 3)
    
    return memory_gb


def format_size(bytes_size: int) -> str:
    """Format byte size into human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if abs(bytes_size) < 1024.0:
            return f"{bytes_size:3.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} TB"


def create_progress_tracker(total_items: int, description: str = "Processing"):
    """Create a simple progress tracker."""
    import sys
    
    class SimpleProgress:
        def __init__(self, total, desc):
            self.total = total
            self.desc = desc
            self.current = 0
            self.last_percent = -1
        
        def update(self, increment: int = 1):
            self.current += increment
            percent = int((self.current / self.total) * 100)
            
            if percent > self.last_percent:
                self.last_percent = percent
                sys.stdout.write(f"\r{self.desc}: {percent}% ({self.current}/{self.total})")
                sys.stdout.flush()
        
        def close(self):
            sys.stdout.write(f"\r{self.desc}: 100% ({self.current}/{self.total})\n")
            sys.stdout.flush()
    
    return SimpleProgress(total_items, description)


def safe_json_load(file_path: str) -> Any:
    """Safely load JSON file with error handling."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {file_path}: {e}")
        return None
    except FileNotFoundError:
        print(f"Error: File not found {file_path}")
        return None
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def safe_json_save(data: Any, file_path: str) -> bool:
    """Safely save data to JSON file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"Error saving to {file_path}: {e}")
        return False


def create_backup(file_path: str, backup_dir: str = "./backups") -> str:
    """Create a backup of the specified file."""
    backup_dir = Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = Path(file_path).stem
    backup_path = backup_dir / f"{timestamp}_backup.json"
    
    return str(backup_path)