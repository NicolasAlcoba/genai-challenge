# GenAI Challenge 

## Overview
This repository provides a retrieval-augmented generation (RAG) agent tailored for aviation knowledge. The system processes PDF documents using FAISS vector search, implements semantic retrieval with sentence transformers, and generates responses via a Hugging Face text generation pipeline. It includes an MCP (Model Context Protocol) server/client architecture for interactive chat, vector store management, and document processing capabilities.

## Getting Started

### Prerequisites
- Python 3.10+
- Docker

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Change to mlops directory
cd mlops

# Build vector store from PDFs
make build-vector-store

# Build Docker image
make build-docker

# Run interactive chat
make run-chat

# Run with debug output
make run-chat DEBUG=1

# See all available make commands
make help
```

#### Example question
```

🔍 Your question: What's a lazy 8?
🤔 Thinking...

📚 Response:
----------------------------------------
A lazy 8 is a flight training maneuver designed to develop the proper coordination of the flight controls across a wide range of airspeeds and attitudes. It is the only standard flight training maneuver in which flight control pressures are constantly changing. In an attempt to simplify the discussion about this maneuver, the lazy 8 can be loosely compared to the ground reference maneuver, S- turns across

📚 Sources: Based on 5 relevant document sections
----------------------------------------
```

## Challenge

This challenge consists of three main tasks and should take 2-4 hours. Feel free to demonstrate your ML engineering skills and approach - we want to see how you think and experiment:

### 1. Implement Evaluation Framework
**File:** `src/evaluation.py`

Design and implement a simple evaluation pipeline that demonstrates your understanding of RAG system assessment. **Focus on implementing at least one evaluation metric thoroughly** rather than trying to cover everything:

- **Metrics**: What would you measure? (relevance, faithfulness, answer quality, retrieval accuracy)
- **Evaluation datasets**: How would you create or curate test questions and ground truth?
- **Automated vs Human evaluation**: What's your strategy for scalable assessment?
- **Experimentation**: How would you A/B test different configurations?

**Implementation expectation**: Choose one metric type (e.g., retrieval accuracy, answer relevance, or faithfulness) and implement it with a small evaluation dataset. Document your approach for the other metrics in `DESIGN_AND_OPS.md`.

```bash
python -m src.main eval
```

### 2. System Improvements & Experimentation
Analyze the current system and implement improvements. Show your ML engineering thought process:

**Prompt Engineering & LLM Usage:**
- Experiment with different prompt strategies for better responses
- Consider few-shot examples, chain-of-thought, or role-based prompting
- How would you handle different types of aviation questions?

**Retrieval & Performance:**
- Analyze retrieval quality and propose improvements
- Consider chunking strategies, embedding models, or re-ranking
- How would you handle edge cases or ambiguous queries?

**Model Selection & Configuration:**
- Experiment with different models or parameters
- Consider cost vs quality trade-offs
- How would you choose between different LLM providers?

Implement at least one improvement and demonstrate your methodology.

### 3. Production Roadmap
Given this POC needs to be productionalized for a university with thousands of student pilots, create a production backlog addressing:
- **Scalability**: How would you handle thousands of concurrent users and requests?
- **Infrastructure**: What would your MLOps pipeline look like?
- **Monitoring & Observability**: How would you track system performance and model drift?
- **Privacy & Safety**: What guardrails and data protection measures are needed?
- **Deployment Strategy**: How would you ensure reliable updates and rollbacks?
- **Usage Control**: How would you prevent misuse while maintaining educational value?

**📋 Documentation Requirement:**
In addition to the code, please answer the following questions in a new file named `DESIGN_AND_OPS.md`. This is as important as the code itself.

## What We're Looking For

This challenge is designed to evaluate your ML engineering capabilities across multiple dimensions:

**🧪 Experimentation Mindset**
- How do you approach problem-solving and hypothesis testing?
- Can you design meaningful experiments and interpret results?
- Do you consider multiple approaches and trade-offs?

**📊 Metrics & Evaluation**
- Understanding of appropriate metrics for RAG systems
- Ability to design evaluation frameworks that matter
- Experience with both automated and human evaluation strategies

**🤖 LLM Engineering & Prompt Design**
- Practical experience with language models and their capabilities
- Understanding of prompt engineering techniques and when to apply them
- Knowledge of model selection criteria and configuration

**🏗️ Production Thinking**
- Ability to translate POCs into scalable systems
- Understanding of MLOps principles and infrastructure needs
- Consideration of real-world constraints and requirements

**📝 Communication**
- Clear documentation of your approach and reasoning
- Ability to explain technical decisions and trade-offs
- Structured thinking about complex problems

Feel free to be creative and show your expertise - we value depth over breadth, so focus on areas where you can demonstrate real understanding.
