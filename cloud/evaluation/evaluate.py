#!/usr/bin/env python3
"""
Comprehensive evaluation and benchmarking script for DeepSeek fine-tuned models.
Supports 30+ benchmarks across coding, reasoning, and general NLP tasks.
"""

import os
import sys
import argparse
import yaml
import json
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import torch
import numpy as np
from tqdm import tqdm
from difflib import SequenceMatcher
import re

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    import evaluate
except ImportError:
    print("Error: Required libraries not installed")
    print("Install with: pip install transformers peft evaluate torch numpy")
    sys.exit(1)


# ==========================================
# BENCHMARK DEFINITIONS AND PROMPTS
# ==========================================

class BenchmarkSuite:
    """Collection of benchmark prompts and evaluation methods."""
    
    @staticmethod
    def get_benchmarks() -> Dict[str, Dict]:
        """Return all available benchmarks."""
        return {
            # CODE GENERATION BENCHMARKS
            "python_basics": {
                "name": "Python Basics",
                "description": "Basic Python syntax and operations",
                "category": "coding",
                "prompts": [
                    "Write a Python function that calculates the factorial of a number recursively.",
                    "Create a Python class representing a Bank with account management methods.",
                    "Write a function to find the maximum value in a list without using built-in max().",
                    "Implement binary search algorithm in Python.",
                    "Create a decorator that measures execution time of functions.",
                    "Write a function to flatten a nested Python list.",
                    "Implement a simple linked list in Python.",
                    "Create a Python generator that yields Fibonacci numbers.",
                    "Write unit tests for a simple calculator class.",
                    "Implement bubble sort algorithm with early termination optimization."
                ],
                "expected_outputs": ["function", "factorial", "recursive"],
                "evaluation_method": "code_quality"
            },
            
            "python_advanced": {
                "name": "Python Advanced",
                "description": "Advanced Python concepts and best practices",
                "category": "coding",
                "prompts": [
                    "Implement a context manager for temporary directory handling.",
                    "Create a metaclass that enforces method implementations in subclasses.",
                    "Write an async/await version of a web scraper.",
                    "Implement a custom exception hierarchy for database operations.",
                    "Create a thread-safe singleton pattern in Python.",
                    "Write a function that uses multiprocessing for parallel computation.",
                    "Implement a memory-efficient data structure for large datasets.",
                    "Create a custom Python iterator for complex data processing.",
                    "Write code using dataclasses with field validators.",
                    "Implement a plugin system using Python's import hooks."
                ],
                "expected_outputs": ["async", "await", "context", "manager"],
                "evaluation_method": "code_quality"
            },
            
            "javascript_basics": {
                "name": "JavaScript Basics",
                "description": "Fundamental JavaScript concepts and syntax",
                "category": "coding",
                "prompts": [
                    "Write a JavaScript function to reverse a string.",
                    "Create a JavaScript promise that resolves after 2 seconds.",
                    "Implement array.filter and array.reduce manually.",
                    "Write a JavaScript class with static methods.",
                    "Create an event listener for a button click with event delegation.",
                    "Write a function to deep clone an object in JavaScript.",
                    "Implement a simple debounce function.",
                    "Create a closure that maintains state across function calls.",
                    "Write async/await code to fetch and process API data.",
                    "Implement a custom error class in JavaScript."
                ],
                "expected_outputs": ["function", "return", "console"],
                "evaluation_method": "code_quality"
            },
            
            "java_programming": {
                "name": "Java Programming",
                "description": "Java language fundamentals and OOP concepts",
                "category": "coding",
                "prompts": [
                    "Write a Java interface with default methods.",
                    "Implement a generic Java class for a stack data structure.",
                    "Create a Java Stream API example for data processing.",
                    "Write a Java exception handling_best practices example.",
                    "Implement a Java singleton pattern using enum.",
                    "Create a Java lambda expression with functional interfaces.",
                    "Write a Java service with dependency injection.",
                    "Implement thread-safe counter using AtomicInteger.",
                    "Create a Java annotation processor example.",
                    "Write a Java utility class for string manipulation."
                ],
                "expected_outputs": ["public", "class", "private", "return"],
                "evaluation_method": "code_quality"
            },
            
            "cpp_programming": {
                "name": "C++ Programming",
                "description": "Modern C++ with STL and best practices",
                "category": "coding",
                "prompts": [
                    "Write a C++ template class for a generic container.",
                    "Implement RAII pattern for resource management.",
                    "Create a C++ lambda expression with captures.",
                    "Write a C++ program using smart pointers.",
                    "Implement a C++ move constructor and assignment operator.",
                    "Create a C++ exception class with custom message.",
                    "Write a C++ function using std::vector and algorithms.",
                    "Implement a C++ thread with synchronization primitives.",
                    "Create a C++ class with operator overloading.",
                    "Write a C++ program using file I/O streams."
                ],
                "expected_outputs": ["class", "public", "private", "std::"],
                "evaluation_method": "code_quality"
            },
            
            "go_programming": {
                "name": "Go Programming",
                "description": "Go language fundamentals and idioms",
                "category": "coding",
                "prompts": [
                    "Write a Go function with multiple return values.",
                    "Implement a Go struct with methods.",
                    "Create a Go channel for concurrent communication.",
                    "Write a Go interface with multiple implementations.",
                    "Implement Go error handling best practices.",
                    "Create a Go goroutine with waitgroup synchronization.",
                    "Write a Go program that uses defer statements.",
                    "Implement a Go HTTP server with handlers.",
                    "Create a Go test file with benchmark functions.",
                    "Write a Go package with public and private functions."
                ],
                "expected_outputs": ["func", "return", "package", "import"],
                "evaluation_method": "code_quality"
            },
            
            "rust_programming": {
                "name": "Rust Programming",
                "description": "Rust language fundamentals and memory safety",
                "category": "coding",
                "prompts": [
                    "Write a Rust struct with impl blocks.",
                    "Implement Rust ownership and borrowing concepts.",
                    "Create a Rust enum with variants.",
                    "Write a Rust function returning Result<T, E>.",
                    "Implement Rust pattern matching with match.",
                    "Create a Rust trait with default implementations.",
                    "Write Rust code using closures and iterators.",
                    "Implement a Rust module system example.",
                    "Create Rust unit tests with assertions.",
                    "Write a Rust program using external crates."
                ],
                "expected_outputs": ["fn", "struct", "let", "mut"],
                "evaluation_method": "code_quality"
            },
            
            # WEB DEVELOPMENT
            "html_css": {
                "name": "HTML/CSS",
                "description": "HTML structure and CSS styling",
                "category": "web",
                "prompts": [
                    "Create an HTML form with proper input types and validation.",
                    "Write CSS flexbox layout for responsive design.",
                    "Implement CSS grid for complex layouts.",
                    "Create a responsive navbar with media queries.",
                    "Write CSS animations and transitions.",
                    "Create accessible HTML semantic elements.",
                    "Implement CSS custom properties (variables).",
                    "Write CSS for card-based layout design.",
                    "Create a login form with CSS styling.",
                    "Implement CSS typography best practices."
                ],
                "expected_outputs": ["<div>", "<style>","css"],
                "evaluation_method": "web_quality"
            },
            
            "react_development": {
                "name": "React Development",
                "description": "React.js component development",
                "category": "web",
                "prompts": [
                    "Create a React functional component with hooks.",
                    "Implement React useEffect for API calls.",
                    "Write a React custom hook for state management.",
                    "Create a React form with controlled inputs.",
                    "Implement React Router navigation.",
                    "Write a React component with TypeScript.",
                    "Create a React context for global state.",
                    "Implement React error boundaries.",
                    "Write a React component using useRef.",
                    "Create a React memo optimization example."
                ],
                "expected_outputs": ["useState", "useEffect", "React"],
                "evaluation_method": "web_quality"
            },
            
            # DATA SCIENCE
            "pandas_analysis": {
                "name": "Pandas Data Analysis",
                "description": "Python pandas for data manipulation",
                "category": "data_science",
                "prompts": [
                    "Load CSV file and display basic statistics with pandas.",
                    "Filter and group data in pandas DataFrame.",
                    "Merge/join multiple pandas DataFrames.",
                    "Handle missing values in pandas.",
                    "Create pivot tables with pandas.",
                    "Apply functions to DataFrame columns.",
                    "Resample time series data with pandas.",
                    "Plot data using pandas plotting functionality.",
                    "Write complex query operations with pandas.query().",
                    "Optimize pandas DataFrame memory usage."
                ],
                "expected_outputs": ["pd.read", "import pandas", "DataFrame"],
                "evaluation_method": "data_quality"
            },
            
            "numpy_computations": {
                "name": "NumPy Computations",
                "description": "NumPy array operations and computations",
                "category": "data_science",
                "prompts": [
                    "Create NumPy arrays with different initialization methods.",
                    "Perform element-wise operations on arrays.",
                    "Implement array broadcasting operations.",
                    "Linear algebra operations with numpy.linalg.",
                    "Random number generation with numpy.random.",
                    "Statistical calculations on arrays.",
                    "Array masking and filtering operations.",
                    "Array reshaping and transposition.",
                    "File I/O operations for NumPy arrays.",
                    "Performance optimization with vectorized operations."
                ],
                "expected_outputs": ["np.array", "import numpy", "shape"],
                "evaluation_method": "data_quality"
            },
            
            # MACHINE LEARNING
            "sklearn_basics": {
                "name": "Scikit-learn Basics",
                "description": "Scikit-learn model training and evaluation",
                "category": "machine_learning",
                "prompts": [
                    "Train a linear regression model with scikit-learn.",
                    "Implement classification with random forest.",
                    "Feature scaling and preprocessing pipeline.",
                    "Cross-validation with cross_val_score.",
                    "Hyperparameter tuning with GridSearchCV.",
                    "Evaluate model performance with metrics.",
                    "Handle imbalanced datasets with resampling.",
                    "Feature selection techniques.",
                    "Pipeline creation for ML workflow.",
                    "Model persistence with joblib."
                ],
                "expected_outputs": ["from sklearn", "fit", "predict", "score"],
                "evaluation_method": "ml_quality"
            },
            
            "tensorflow_basics": {
                "name": "TensorFlow Basics",
                "description": "Deep learning with TensorFlow/Keras",
                "category": "machine_learning",
                "prompts": [
                    "Build a simple neural network with TensorFlow.",
                    "Create custom loss function for TensorFlow.",
                    "Implement TensorFlow data pipeline with tf.data.",
                    "Train model with custom callbacks.",
                    "Transfer learning with pre-trained models.",
                    "Create TensorFlow layers and activations.",
                    "Implement TensorFlow metrics for monitoring.",
                    "Build RNN model with TensorFlow.",
                    "Image data preprocessing for TensorFlow.",
                    "TensorFlow model deployment and serving."
                ],
                "expected_outputs": ["tf.keras", "Model", "compile"],
                "evaluation_method": "ml_quality"
            },
            
            # ALGORITHMS
            "algorithms_sorting": {
                "name": "Sorting Algorithms",
                "description": "Various sorting algorithm implementations",
                "category": "algorithms",
                "prompts": [
                    "Implement quicksort algorithm with proper partitioning.",
                    "Create merge sort with divide and conquer approach.",
                    "Heapsort implementation with heap operations.",
                    "Radix sort for integer sorting.",
                    "Bucket sort for uniformly distributed data.",
                    "Implement stable sort for custom objects.",
                    "Compare sorting algorithms with time complexity.",
                    "Optimize sorting for nearly sorted arrays.",
                    "Implement external sort for large files.",
                    "Parallel sorting with divide and conquer."
                ],
                "expected_outputs": ["sort", "algorithm", "complexity"],
                "evaluation_method": "algorithm_quality"
            },
            
            "algorithms_searching": {
                "name": "Searching Algorithms",
                "description": "Search algorithm implementations",
                "category": "algorithms",
                "prompts": [
                    "Implement binary search with edge case handling.",
                    "Create depth-first search (DFS) for graphs.",
                    "Breadth-first search (BFS) implementation.",
                    "A* search algorithm with heuristics.",
                    "Implement binary search tree operations.",
                    "Hash table collision resolution strategies.",
                    "Implement KMP string matching algorithm.",
                    "Regular expression pattern matching.",
                    "Search in rotated sorted array.",
                    "Find nearest neighbors with efficient algorithms."
                ],
                "expected_outputs": ["search", "binary", "algorithm"],
                "evaluation_method": "algorithm_quality"
            },
            
            "dynamic_programming": {
                "name": "Dynamic Programming",
                "description": "DP problems and solutions",
                "category": "algorithms",
                "prompts": [
                    "Solve knapsack problem with dynamic programming.",
                    "Implement memoization for recursive problems.",
                    "Longest common subsequence solution.",
                    "Optimal binary search tree construction.",
                    "Matrix chain multiplication optimization.",
                    "Coin change problem with DP.",
                    "Edit distance calculation.",
                    "Subset sum problem solution.",
                    "Catalan numbers computation.",
                    "DP solution for stock trading problem."
                ],
                "expected_outputs": ["dynamic", "programming", "memo", "dp"],
                "evaluation_method": "algorithm_quality"
            },
            
            # API DEVELOPMENT
            "api_development": {
                "name": "API Development",
                "description": "RESTful API design and implementation",
                "category": "api",
                "prompts": [
                    "Design RESTful API endpoints for user management.",
                    "Implement API authentication with JWT.",
                    "Create API versioning strategy.",
                    "Handle API errors and status codes properly.",
                    "Implement API rate limiting middleware.",
                    "API documentation with OpenAPI/Swagger.",
                    "Design pagination for API responses.",
                    "API testing strategies and tools.",
                    "Implement API caching mechanisms.",
                    "Secure API development best practices."
                ],
                "expected_outputs": ["REST", "API", "endpoint"],
                "evaluation_method": "api_quality"
            },
            
            "database_operations": {
                "name": "Database Operations",
                "description": "SQL and database management",
                "category": "database",
                "prompts": [
                    "Write SQL queries for INSERT, UPDATE, DELETE operations.",
                    "Create indexed views for performance.",
                    "Implement database transactions with ACID properties.",
                    "Database normalization principles and examples.",
                    "Write complex JOIN queries with multiple tables.",
                    "Create stored procedures and functions.",
                    "Database backup and restore procedures.",
                    "Implement database replication strategies.",
                    "Query optimization and execution plans.",
                    "Design database schema for e-commerce system."
                ],
                "expected_outputs": ["SELECT", "FROM", "WHERE", "JOIN"],
                "evaluation_method": "database_quality"
            },
            
            # GENERAL TASKS
            "text_processing": {
                "name": "Text Processing",
                "description": "Natural language text manipulation",
                "category": "general",
                "prompts": [
                    "Clean and preprocess text data for analysis.",
                    "Implement sentiment analysis on text.",
                    "Extract entities from unstructured text.",
                    "Text classification with machine learning.",
                    "Language detection algorithms.",
                    "Text similarity measurement techniques.",
                    "Implement text summarization.",
                    "Named entity recognition implementation.",
                    "Tokenization and stemming methods.",
                    "Handle multilingual text processing."
                ],
                "expected_outputs": ["text", "process", "language"],
                "evaluation_method": "text_quality"
            },
            
            "file_operations": {
                "name": "File Operations",
                "description": "File system operations and handling",
                "category": "general",
                "prompts": [
                    "Read and write files in different formats.",
                    "Implement file upload and download functionality.",
                    "Handle file permissions and access control.",
                    "Compress and decompress files.",
                    "File system traversal and searching.",
                    "Implement file backup and versioning.",
                    "Handle large file processing efficiently.",
                    "File format conversion utilities.",
                    "Directory monitoring and event handling.",
                    "Secure file handling and cleanup."
                ],
                "expected_outputs": ["file", "write", "read", "open"],
                "evaluation_method": "general_quality"
            },
            
            "network_programming": {
                "name": "Network Programming",
                "description": "Socket programming and network operations",
                "category": "general",
                "prompts": [
                    "Create TCP socket server and client.",
                    "Implement HTTP client with connection pooling.",
                    "Handle network timeouts and retries.",
                    "Implement WebSocket communication.",
                    "Network error handling and recovery.",
                    "Secure socket programming with SSL/TLS.",
                    "Network protocol parsing and validation.",
                    "Implement load balancing for network requests.",
                    "Network security best practices.",
                    "Handle network congestion and flow control."
                ],
                "expected_outputs": ["socket", "network", "connect", "server"],
                "evaluation_method": "general_quality"
            },
            
            "security_implementation": {
                "name": "Security Implementation",
                "description": "Security best practices and implementations",
                "category": "general",
                "prompts": [
                    "Implement password hashing and verification.",
                    "Create secure session management.",
                    "Input validation and sanitization.",
                    "SQL injection prevention techniques.",
                    "Cross-site scripting (XSS) protection.",
                    "Implement content security policy.",
                    "Secure random number generation.",
                    "Cryptography basics and implementation.",
                    "Authentication and authorization design.",
                    "Security audit logging and monitoring."
                ],
                "expected_outputs": ["security", "encrypt", "hash", "protect"],
                "evaluation_method": "security_quality"
            },
            
            "performance_optimization": {
                "name": "Performance Optimization",
                "description": "Code performance improvements",
                "category": "general",
                "prompts": [
                    "Identify and fix performance bottlenecks.",
                    "Implement caching strategies for optimization.",
                    "Memory management and leak prevention.",
                    "Database query optimization techniques.",
                    "Algorithm selection for performance.",
                    "Parallel processing for speed improvement.",
                    "Code profiling and analysis.",
                    "Optimize network request handling.",
                    "Lazy loading implementation.",
                    "Performance monitoring and metrics."
                ],
                "expected_outputs": ["optimize", "performance", "cache", "memory"],
                "evaluation_method": "general_quality"
            },
            
            "error_handling": {
                "name": "Error Handling",
                "description": "Robust error handling and recovery",
                "category": "general",
                "prompts": [
                    "Implement custom exception classes.",
                    "Create error handling middleware.",
                    "Design error logging system.",
                    "Handle edge cases in algorithms.",
                    "Implement retry logic with exponential backoff.",
                    "Error recovery and fallback mechanisms.",
                    "Validation error handling patterns.",
                    "Network error handling strategies.",
                    "Database connection error management.",
                    "User-friendly error messages design."
                ],
                "expected_outputs": ["error", "exception", "catch", "handle"],
                "evaluation_method": "general_quality"
            },
            
            "testing_strategies": {
                "name": "Testing Strategies",
                "description": "Software testing methodologies",
                "category": "general",
                "prompts": [
                    "Write unit tests for business logic.",
                    "Implement integration testing framework.",
                    "Create mock objects for testing.",
                    "Design test data generation strategies.",
                    "Performance testing implementation.",
                    "Load testing for web applications.",
                    "User interface testing automation.",
                    "Test-driven development examples.",
                    "Continuous integration testing pipelines.",
                    "Test coverage analysis and improvement."
                ],
                "expected_outputs": ["test", "assert", "mock", "expect"],
                "evaluation_method": "general_quality"
            },
            
            "devops_tasks": {
                "name": "DevOps Tasks",
                "description": "DevOps and automation tasks",
                "category": "general",
                "prompts": [
                    "Write Dockerfiles for application containers.",
                    "Create Kubernetes deployment configuration.",
                    "Implement CI/CD pipeline scripts.",
                    "Infrastructure as code with Terraform.",
                    "Configuration management with Ansible.",
                    "Monitoring and alerting setup.",
                    "Log aggregation and analysis.",
                    "Automated deployment strategies.",
                    "Environment variable management.",
                    "Rollback procedures for deployments."
                ],
                "expected_outputs": ["deploy", "build", "pipeline", "monitor"],
                "evaluation_method": "general_quality"
            },
            
            "code_documentation": {
                "name": "Code Documentation",
                "description": "Documentation and code comments",
                "category": "general",
                "prompts": [
                    "Write comprehensive docstrings for functions.",
                    "Create API documentation examples.",
                    "Document software architecture decisions.",
                    "Write user guides and tutorials.",
                    "Generate documentation from code comments.",
                    "Create diagrams for system documentation.",
                    "Document code changes and release notes.",
                    "Setup automated documentation generation.",
                    "Write inline comments for complex logic.",
                    "Maintain README and project documentation."
                ],
                "expected_outputs": ["document", "explain", "describe", "example"],
                "evaluation_method": "general_quality"
            },
            
            "code_review": {
                "name": "Code Review",
                "description": "Code review best practices",
                "category": "general",
                "prompts": [
                    "Identify code smells and refactoring opportunities.",
                    "Review for security vulnerabilities.",
                    "Check code for performance issues.",
                    "Verify code compliance with standards.",
                    "Suggest improvements for code readability.",
                    "Review error handling completeness.",
                    "Check for proper resource cleanup.",
                    "Review test coverage suggestions.",
                    "Identify potential bugs and edge cases.",
                    "Provide constructive code review comments."
                ],
                "expected_outputs": ["review", "improve", "suggest", "fix"],
                "evaluation_method": "general_quality"
            },
            
            "software_design": {
                "name": "Software Design",
                "description": "Software architecture and design patterns",
                "category": "general",
                "prompts": [
                    "Implement singleton design pattern.",
                    "Create factory pattern implementation.",
                    "Observer pattern for event handling.",
                    "Strategy pattern for algorithm selection.",
                    "Builder pattern for complex objects.",
                    "Decorator pattern for functionality extension.",
                    "Dependency injection implementation.",
                    "MVC architecture design.",
                    "Microservices design principles.",
                    "API design patterns and best practices."
                ],
                "expected_outputs": ["pattern", "design", "class", "interface"],
                "evaluation_method": "design_quality"
            },
            
            "git_workflow": {
                "name": "Git Workflow",
                "description": "Git version control workflows",
                "category": "general",
                "prompts": [
                    "Implement Git branching strategy.",
                    "Write Git commit message guidelines.",
                    "Resolve merge conflicts in Git.",
                    "Git workflow for team collaboration.",
                    "Git hooks for automation.",
                    "Release management with Git.",
                    "Git bisect for bug hunting.",
                    "Git stash and cherry-pick scenarios.",
                    "Git submodules usage.",
                    "Git best practices and conventions."
                ],
                "expected_outputs": ["git", "branch", "commit", "merge"],
                "evaluation_method": "general_quality"
            }
        }


class ModelEvaluator:
    """Main evaluation class for benchmarking models."""
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = {}
        
        # Load model and tokenizer
        self.model, self.tokenizer = self.load_model()
        if self.model is None:
            raise ValueError("Failed to load model")
    
    def load_model(self):
        """Load model and tokenizer."""
        try:
            io_config = self.config['cloud']['io']
            model_config = self.config['model']
            adapter_path = io_config.get('adapter_dir', './adapters')
            
            self.logger.info("Loading model and tokenizer...")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_config['base_model'],
                cache_dir=io_config['cache_dir'],
                trust_remote_code=True
            )
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                model_config['base_model'],
                torch_dtype=torch.float16,
                device_map="auto",
                cache_dir=io_config['cache_dir'],
                trust_remote_code=True
            )
            
            # Load adapter if exists
            if os.path.exists(adapter_path):
                self.logger.info(f"Loading adapter from {adapter_path}")
                model = PeftModel.from_pretrained(base_model, adapter_path)
            else:
                self.logger.warning("No adapter found, using base model")
                model = base_model
            
            self.logger.info("✓ Model and tokenizer loaded successfully")
            return model, tokenizer
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return None, None
    
    def generate_response(self, prompt: str, max_new_tokens: int = 512) -> str:
        """Generate response from model."""
        try:
            # Format prompt
            formatted_prompt = f"<|User|>: {prompt}\n<|Assistant|>:"
            
            # Tokenize
            inputs = self.tokenizer.encode(formatted_prompt, return_tensors='pt').to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    top_p=0.95,
                    do_sample=True,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
            
            return response.strip()
            
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return f"Error: {str(e)}"
    
    def evaluate_response_quality(self, response: str, benchmark_type: str, expected_keywords: List[str]) -> Dict[str, Any]:
        """Evaluate response quality based on benchmark type."""
        quality_score = 0.0
        details = {}
        
        # Basic length check
        if len(response) > 50:
            quality_score += 0.2
        details["length_adequate"] = len(response) > 50
        
        # Check for expected keywords
        keyword_matches = 0
        for keyword in expected_keywords:
            if keyword.lower() in response.lower():
                keyword_matches += 1
        
        keyword_score = keyword_matches / len(expected_keywords) if expected_keywords else 0.5
        quality_score += (keyword_score * 0.4)
        details["keyword_coverage"] = keyword_matches
        details["expected_keywords"] = expected_keywords
        
        # Technical quality based on benchmark type
        if benchmark_type == "coding":
            quality_score += self._evaluate_coding_quality(response) * 0.4
        elif benchmark_type == "web":
            quality_score += self._evaluate_web_quality(response) * 0.4
        elif benchmark_type == "data_science":
            quality_score += self._evaluate_data_quality(response) * 0.4
        elif benchmark_type == "machine_learning":
            quality_score += self._evaluate_ml_quality(response) * 0.4
        elif benchmark_type == "algorithms":
            quality_score += self._evaluate_algorithm_quality(response) * 0.4
        elif benchmark_type == "api":
            quality_score += self._evaluate_api_quality(response) * 0.4
        elif benchmark_type == "database":
            quality_score += self._evaluate_database_quality(response) * 0.4
        elif benchmark_type == "general":
            quality_score += self._evaluate_general_quality(response) * 0.4
        else:
            quality_score += 0.2  # Default score
        
        details["final_score"] = min(quality_score, 1.0)
        return details
    
    def _evaluate_coding_quality(self, response: str) -> float:
        """Evaluate coding-specific quality."""
        score = 0.5
        
        # Check for code blocks
        if "```" in response or "def " in response or "function" in response:
            score += 0.2
        
        # Check for proper structure
        if any(indent in response for indent in ["    ", "\t"]):
            score += 0.15
        
        # Check for comments
        if "#" in response or "//" in response or "/*" in response:
            score += 0.15
        
        return min(score, 1.0)
    
    def _evaluate_web_quality(self, response: str) -> float:
        """Evaluate web development quality."""
        score = 0.5
        
        if any(tag in response for tag in ["<div>", "<style>", "<script>", "React", "Vue"]):
            score += 0.25
        
        if "CSS" in response or "HTML" in response or "JavaScript" in response:
            score += 0.25
        
        return min(score, 1.0)
    
    def _evaluate_data_quality(self, response: str) -> float:
        """Evaluate data science quality."""
        score = 0.5
        
        if any(lib in response for lib in ["pandas", "numpy", "Data", "frame"]):
            score += 0.25
        
        if code := response.split('\n'):
            if any("import" in line for line in code):
                score += 0.25
        
        return min(score, 1.0)
    
    def _evaluate_ml_quality(self, response: str) -> float:
        """Evaluate machine learning quality."""
        score = 0.5
        
        if any(term in response for term in ["train", "model", "predict", "fit", "sklearn", "TF"]):
            score += 0.25
        
        if "accuracy" in response or "loss" in response or "eval" in response:
            score += 0.25
        
        return min(score, 1.0)
    
    def _evaluate_algorithm_quality(self, response: str) -> float:
        """Evaluate algorithm quality."""
        score = 0.5
        
        if any(term in response for term in ["complexity", "O(n)", "algorithm", "sort", "search"]):
            score += 0.3
        
        if any(control in response for control in ["for", "if", "while", "recurs"]):
            score += 0.2
        
        return min(score, 1.0)
    
    def _evaluate_api_quality(self, response: str) -> float:
        """Evaluate API quality."""
        score = 0.5
        
        if any(term in response for term in ["endpoint", "request", "response", "HTTP", "API"]):
            score += 0.3
        
        return min(score, 1.0)
    
    def _evaluate_database_quality(self, response: str) -> float:
        """Evaluate database quality."""
        score = 0.5
        
        if any(term in response for term in ["SELECT", "FROM", "table", "database", "SQL"]):
            score += 0.3
        
        return min(score, 1.0)
    
    def _evaluate_general_quality(self, response: str) -> float:
        """Evaluate general task quality."""
        score = 0.5
        
        # Check for structured response
        if len(response.split('\n')) > 2:
            score += 0.25
        
        # Check for explanatory content
        if any(explain in response.lower() for explain in ["because", "therefore", "example", "first"]):
            score += 0.25
        
        return min(score, 1.0)
    
    def run_benchmark(self, benchmark_name: str, benchmark_data: Dict[str, Any], max_prompts: int = None) -> Dict[str, Any]:
        """Run a single benchmark evaluation."""
        self.logger.info(f"Running benchmark: {benchmark_name}")
        
        prompts = benchmark_data['prompts']
        if max_prompts:
            prompts = prompts[:max_prompts]
        
        results = {
            "benchmark_name": benchmark_name,
            "description": benchmark_data['description'],
            "category": benchmark_data['category'],
            "prompts_tested": len(prompts),
            "responses": [],
            "quality_scores": [],
            "average_quality": 0.0,
            "execution_time": 0.0
        }
        
        start_time = time.time()
        
        for i, prompt in enumerate(tqdm(prompts, desc=f"Testing {benchmark_name}", leave=False)):
            try:
                response = self.generate_response(prompt)
                quality_eval = self.evaluate_response_quality(
                    response, 
                    benchmark_data['category'],
                    benchmark_data.get('expected_outputs', [])
                )
                
                result_item = {
                    "prompt": prompt,
                    "response": response,
                    "response_length": len(response),
                    "quality_score": quality_eval["final_score"],
                    "evaluation_details": quality_eval
                }
                
                results["responses"].append(result_item)
                results["quality_scores"].append(quality_eval["final_score"])
                
            except Exception as e:
                self.logger.error(f"Error processing prompt {i}: {e}")
                results["responses"].append({
                    "error": str(e),
                    "quality_score": 0.0
                })
        
        results["execution_time"] = time.time() - start_time
        results["average_quality"] = np.mean(results["quality_scores"]) if results["quality_scores"] else 0.0
        
        self.logger.info(f"✓ Benchmark {benchmark_name} completed. Avg quality: {results['average_quality']:.2f}")
        
        return results
    
    def run_all_benchmarks(self, max_prompts_per_benchmark: int = None) -> Dict[str, Any]:
        """Run all available benchmarks."""
        benchmarks = BenchmarkSuite.get_benchmarks()
        
        all_results = {
            "evaluator_version": "1.0.0",
            "evaluation_date": datetime.now().isoformat(),
            "model_info": {
                "base_model": self.config['model']['base_model'],
                "adapter_dir": self.config['cloud']['io'].get('adapter_dir', './adapters'),
                "device": str(self.device)
            },
            "summary": {
                "total_benchmarks": len(benchmarks),
                "completed_benchmarks": 0,
                "failed_benchmarks": 0,
                "overall_average_quality": 0.0,
                "category_averages": {}
            },
            "benchmark_results": {}
        }
        
        self.logger.info(f"Starting comprehensive evaluation of {len(benchmarks)} benchmarks")
        
        for benchmark_name, benchmark_data in benchmarks.items():
            try:
                benchmark_result = self.run_benchmark(benchmark_name, benchmark_data, max_prompts_per_benchmark)
                all_results["benchmark_results"][benchmark_name] = benchmark_result
                all_results["summary"]["completed_benchmarks"] += 1
            except Exception as e:
                self.logger.error(f"Failed to run benchmark {benchmark_name}: {e}")
                all_results["summary"]["failed_benchmarks"] += 1
        
        # Calculate summary statistics
        quality_scores = [
            result["average_quality"] 
            for result in all_results["benchmark_results"].values()
        ]
        
        if quality_scores:
            all_results["summary"]["overall_average_quality"] = np.mean(quality_scores)
            all_results["summary"]["quality_variance"] = np.var(quality_scores)
            all_results["summary"]["best_benchmark"] = max(
                all_results["benchmark_results"].items(), 
                key=lambda x: x[1]["average_quality"]
            )[0]
            all_results["summary"]["worst_benchmark"] = min(
                all_results["benchmark_results"].items(), 
                key=lambda x: x[1]["average_quality"]
            )[0]
        
        # Calculate category averages
        for result in all_results["benchmark_results"].values():
            category = result["category"]
            if category not in all_results["summary"]["category_averages"]:
                all_results["summary"]["category_averages"][category] = []
            all_results["summary"]["category_averages"][category].append(result["average_quality"])
        
        for category, scores in all_results["summary"]["category_averages"].items():
            all_results["summary"]["category_averages"][category] = np.mean(scores) if scores else 0.0
        
        return all_results
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save evaluation results to JSON file."""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            self.logger.info(f"✓ Results saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate DeepSeek model with comprehensive benchmarks')
    parser.add_argument('--config', type=str, default='config/cloud.yaml',
                        help='Configuration file to use')
    parser.add_argument('--adapter', type=str, default=None,
                        help='Path to adapter (overrides config)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for results')
    parser.add_argument('--benchmarks', type=str, nargs='+', default=None,
                        help='Specific benchmarks to run')
    parser.add_argument('--max-prompts', type=int, default=None,
                        help='Maximum prompts per benchmark')
    parser.add_argument('--benchmark-list', action='store_true',
                        help='List all available benchmarks')
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(__file__).parent.parent / "config" / "cloud.yaml"
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        return 1
    
    # Override adapter path if provided
    if args.adapter:
        config['cloud']['io']['adapter_dir'] = args.adapter
    
    # Setup logging
    log_dir = config['cloud']['io']['log_dir']
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'evaluation.log'), encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    
    # List benchmarks if requested
    if args.benchmark_list:
        benchmarks = BenchmarkSuite.get_benchmarks()
        print("Available benchmarks:")
        for name, data in benchmarks.items():
            print(f"  {name}: {data['description']} ({data['category']})")
        return 0
    
    print("=" * 60)
    print("DeepSeek Model Evaluation")
    print("=" * 60)
    print(f"Configuration: {args.config}")
    print(f"Max prompts per benchmark: {args.max_prompts}")
    print(f"Specific benchmarks: {args.benchmarks}")
    
    # Create evaluator
    evaluator = ModelEvaluator(config, logger)
    
    # Run benchmarks
    if args.benchmarks:
        # Run specific benchmarks
        benchmarks_suite = BenchmarkSuite.get_benchmarks()
        results = {
            "evaluator_version": "1.0.0",
            "evaluation_date": datetime.now().isoformat(),
            "model_info": {
                "base_model": config['model']['base_model'],
                "adapter_dir": config['cloud']['io'].get('adapter_dir', './adapters')
            },
            "summary": {
                "requested_benchmarks": args.benchmarks,
                "completed_benchmarks": 0,
                "overall_average_quality": 0.0
            },
            "benchmark_results": {}
        }
        
        for benchmark_name in args.benchmarks:
            if benchmark_name in benchmarks_suite:
                try:
                    benchmark_result = evaluator.run_benchmark(
                        benchmark_name, 
                        benchmarks_suite[benchmark_name], 
                        args.max_prompts
                    )
                    results["benchmark_results"][benchmark_name] = benchmark_result
                    results["summary"]["completed_benchmarks"] += 1
                except Exception as e:
                    logger.error(f"Failed to run benchmark {benchmark_name}: {e}")
    else:
        # Run all benchmarks
        results = evaluator.run_all_benchmarks(args.max_prompts)
    
    # Save results
    output_path = args.output or os.path.join(Path(__file__).parent.parent / "results", "evaluation_results.json")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    evaluator.save_results(results, output_path)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(f"Total benchmarks: {results['summary']['total_benchmarks'] if 'total_benchmarks' in results['summary'] else len(results['benchmark_results'])}")
    print(f"Completed benchmarks: {results['summary']['completed_benchmarks']}")
    print(f"Overall average quality: {results['summary']['overall_average_quality']:.2f}")
    
    if 'category_averages' in results['summary']:
        print("\nCategory Performance:")
        for category, avg_score in sorted(results['summary']['category_averages'].items(), 
                                        key=lambda x: x[1], reverse=True):
            print(f"  {category}: {avg_score:.2f}")
    
    print(f"\nResults saved to: {output_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())