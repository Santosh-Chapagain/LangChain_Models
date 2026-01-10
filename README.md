# LangChain Models

A comprehensive project demonstrating various LangChain integrations with multiple Large Language Models (LLMs), Chat Models, Embedding Models, Prompt Engineering, and Structured Output handling.

## 📋 Project Overview

This project showcases practical implementations of LangChain with different AI providers including:
- **OpenAI (GPT models)**
- **Anthropic (Claude)**
- **Google Generative AI (Gemini)**
- **Hugging Face Models (local and remote)**
- **Advanced Prompt Engineering**
- **Structured Output with Pydantic and TypedDict**

## 📁 Project Structure

```
LangChainModel/
├── 1.LLMs/
│   └── 1_llm_demo.py                           # Basic LLM integration with OpenAI
├── 2.ChatModels/
│   ├── 1_Chatmodel_openAI.py                   # OpenAI Chat Model implementation
│   ├── 2_Chatmodel_Antropic.py                 # Anthropic Claude integration
│   ├── 3_Chatmodel_google.py                   # Google Generative AI integration
│   ├── 4_Chatmodel_huggingface.ipynb           # Hugging Face Chat Model
│   └── 5_chatmodel_hf_local.py                 # Local Hugging Face model
├── 3.EmbeeddedModels/
│   ├── 1_Embedding_HF_query.py                 # Hugging Face embeddings
│   └── 2_document_similarity.py                # Document similarity using embeddings
├── 4_Prompts/
│   ├── prompts.py                              # Streamlit Research Paper Summarizer
│   ├── chatbot.py                              # Chat bot with conversation history
│   ├── messages.py                             # Basic message handling demo
│   ├── message_placeholder.py                  # Placeholder for chat history
│   ├── chat_prompt_tempelate.py                # Dynamic chat prompt templates
│   ├── prompts_generator.py                    # Prompt template generator
│   ├── templete.json                           # Saved prompt template
│   └── chat_history.txt                        # Chat history storage
├── 5_Structured_Output/
│   ├── pydantic_demo.py                        # Pydantic basics demo
│   ├── typedict_demo.py                        # TypedDict basics demo
│   ├── with_structured_output_pydantic.py      # Structured output with Pydantic
│   └── with_structured_output_typedict.py      # Structured output with TypedDict
├── requirement.txt                              # Project dependencies
├── test.py                                      # Testing script
└── README.md                                    # This file
```

## 🚀 Features

### 1. **LLMs (Large Language Models)**
- Interactive question-answering with GPT-3.5 Turbo
- Real-time response handling with 5-second delay between requests
- Basic LLM invocation and response handling

### 2. **Chat Models**
Multiple implementations with different providers:
- **OpenAI**: GPT models for conversational AI
- **Anthropic**: Claude 3.5 Sonnet for advanced reasoning
- **Google Generative AI**: Gemini 2.5 Flash for versatile tasks
- **Hugging Face**: Both cloud-based and local model deployments (TinyLlama)

### 3. **Embedding Models**
- Document embedding generation using Hugging Face Sentence Transformers
- Document similarity comparison using cosine similarity
- Vector-based semantic search capabilities
- Real-world example: Finding most relevant document from a corpus

### 4. **Prompt Engineering**
- **Message Types**: SystemMessage, HumanMessage, AIMessage
- **Chat Templates**: Dynamic prompt templates with variables
- **Message Placeholders**: Storing and managing chat history
- **Template Persistence**: Saving and loading prompt templates from JSON
- **Interactive Chatbot**: Full conversational AI with memory
- **Streamlit Application**: Research paper summarizer with customizable styles and lengths

### 5. **Structured Output**
- **Pydantic Models**: Type-safe data validation with BaseModel
- **TypedDict**: Lightweight type annotations for dictionaries
- **LLM Integration**: Extracting structured data from unstructured text
- **Real-world Use Case**: Product review analysis with sentiment, pros/cons extraction

## 📦 Installation

### Prerequisites
- Python 3.8+
- Virtual Environment (recommended)

### Setup Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Santosh-Chapagain/LangChain_Models.git
   cd LangChainModel
   ```

2. **Create and activate virtual environment:**
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   
   # Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirement.txt
   ```

4. **Configure environment variables:**
   - Create a `.env` file in the project root
   - Add your API keys:
     ```
     OPENAI_API_KEY=your_openai_key
     ANTHROPIC_API_KEY=your_anthropic_key
     GOOGLE_API_KEY=your_google_key
     HUGGINGFACE_API_KEY=your_huggingface_key
     ```

## 📚 Dependencies

- **langchain** - Core LangChain framework
- **langchain-core** - Core abstractions
- **langchain-openai** - OpenAI integration
- **langchain-anthropic** - Anthropic integration
- **langchain-google-genai** - Google Generative AI integration
- **langchain-huggingface** - Hugging Face integration
- **transformers** - Transformers library for HF models
- **huggingface-hub** - Hugging Face Hub access
- **openai** - OpenAI client library
- **google-generativeai** - Google Generative AI client
- **python-dotenv** - Environment variable management
- **pydantic** - Data validation and settings management
- **scikit-learn** - Machine learning utilities (for cosine similarity)
- **streamlit** - Web app framework for interactive demos
- **numpy** - Numerical computing library
  

## 🎯 Usage Examples

### 1. Running LLM Demo
```bash
python 1.LLMs/1_llm_demo.py
```
Interactive CLI for asking questions to GPT-3.5 Turbo.

### 2. Using Chat Models

**OpenAI Chat:**
```bash
python 2.ChatModels/1_Chatmodel_openAI.py
```

**Anthropic Claude:**
```bash
python 2.ChatModels/2_Chatmodel_Antropic.py
```

**Google Gemini:**
```bash
python 2.ChatModels/3_Chatmodel_google.py
```

**Local Hugging Face Model:**
```bash
python 2.ChatModels/5_chatmodel_hf_local.py
```

### 3. Working with Embeddings

**Generate Embeddings:**
```bash
python 3.EmbeeddedModels/1_Embedding_HF_query.py
```

**Document Similarity Search:**
```bash
python 3.EmbeeddedModels/2_document_similarity.py
```

### 4. Prompt Engineering Examples

**Basic Messages:**
```bash
python 4_Prompts/messages.py
```

**Interactive Chatbot with History:**
```bash
python 4_Prompts/chatbot.py
```

**Chat Prompt Templates:**
```bash
python 4_Prompts/chat_prompt_tempelate.py
```

**Message Placeholder Demo:**
```bash
python 4_Prompts/message_placeholder.py
```

**Research Paper Summarizer (Streamlit App):**
```bash
streamlit run 4_Prompts/prompts.py
```

**Generate Prompt Template:**
```bash
python 4_Prompts/prompts_generator.py
```

### 5. Structured Output Examples

**Pydantic Basics:**
```bash
python 5_Structured_Output/pydantic_demo.py
```

**TypedDict Basics:**
```bash
python 5_Structured_Output/typedict_demo.py
```

**Structured Output with Pydantic (Review Analysis):**
```bash
python 5_Structured_Output/with_structured_output_pydantic.py
```

**Structured Output with TypedDict:**
```bash
python 5_Structured_Output/with_structured_output_typedict.py
```

## 🔑 API Keys Required

| Service | File | Environment Variable |
|---------|------|----------------------|
| OpenAI | `.env` | `OPENAI_API_KEY` |
| Anthropic | `.env` | `ANTHROPIC_API_KEY` |
| Google | `.env` | `GOOGLE_API_KEY` |
| Hugging Face | `.env` | `HUGGINGFACE_API_KEY` |

## 🛠️ Configuration

All configurations are managed through:
- `.env` file for sensitive keys
- Streamlit app settings in `Prompts/prompts.py`
- Model parameters in individual scripts

## 📝 File Descriptions

### 1.LLMs/
| File | Purpose | Key Features |
|------|---------|--------------|
| `1_llm_demo.py` | Interactive LLM chat interface | Simple Q&A loop with GPT-3.5 Turbo Instruct |

### 2.ChatModels/
| File | Purpose | Key Features |
|------|---------|--------------|
| `1_Chatmodel_openAI.py` | OpenAI chat implementation | Interactive chat with GPT models |
| `2_Chatmodel_Antropic.py` | Claude model integration | One-shot question answering with Claude 3.5 Sonnet |
| `3_Chatmodel_google.py` | Google Gemini integration | Interactive chat with Gemini 2.5 Flash |
| `4_Chatmodel_huggingface.ipynb` | Jupyter notebook for HF models | Notebook-based experimentation |
| `5_chatmodel_hf_local.py` | Local Hugging Face models | Run TinyLlama locally without API |

### 3.EmbeeddedModels/
| File | Purpose | Key Features |
|------|---------|--------------|
| `1_Embedding_HF_query.py` | Text embedding generation | Generate vectors for text documents |
| `2_document_similarity.py` | Semantic document comparison | Find most similar documents using cosine similarity |

### 4_Prompts/
| File | Purpose | Key Features |
|------|---------|--------------|
| `prompts.py` | Research Paper Summarizer | Streamlit app with customizable summarization styles |
| `chatbot.py` | Conversational AI with memory | Full chatbot with conversation history tracking |
| `messages.py` | Basic message handling | SystemMessage, HumanMessage, AIMessage demo |
| `message_placeholder.py` | Chat history management | Load and use historical conversations |
| `chat_prompt_tempelate.py` | Dynamic prompt templates | Create reusable prompt templates with variables |
| `prompts_generator.py` | Template generator | Generate and save complex prompt templates |
| `templete.json` | Saved prompt template | Persistent storage of prompt configurations |
| `chat_history.txt` | Chat history storage | Text file for storing conversation logs |

### 5_Structured_Output/
| File | Purpose | Key Features |
|------|---------|--------------|
| `pydantic_demo.py` | Pydantic basics | Simple Pydantic BaseModel example |
| `typedict_demo.py` | TypedDict basics | TypedDict usage demonstration |
| `with_structured_output_pydantic.py` | Product review analyzer | Extract structured data (sentiment, pros/cons) from reviews using Pydantic |
| `with_structured_output_typedict.py` | Review analyzer with TypedDict | Same functionality using TypedDict instead of Pydantic |

## � Detailed Feature Breakdown

### Prompt Engineering Capabilities
1. **Message Types**
   - SystemMessage: Define AI behavior and persona
   - HumanMessage: User inputs
   - AIMessage: AI responses for conversation context

2. **Chat Templates**
   - Dynamic variable substitution
   - Reusable prompt structures
   - Template persistence with JSON

3. **Research Paper Summarizer Features**
   - Multiple explanation styles: Beginner-Friendly, Technical, Code-Oriented, Mathematical
   - Variable length outputs: Short, Medium, Long
   - Popular papers: Attention Is All You Need, BERT, GPT-3, Diffusion Models
   - Structured output with headings and formatting

### Structured Output Capabilities
1. **Schema Definitions**
   - Pydantic: Runtime validation with BaseModel
   - TypedDict: Lightweight type annotations
   - Field descriptions and annotations

2. **Review Analysis Schema**
   - Key themes extraction (list)
   - Summary generation (string)
   - Sentiment analysis (Literal: pos/neg)
   - Pros extraction (optional list)
   - Cons extraction (optional list)
   - Reviewer name extraction (optional string)

3. **Use Cases**
   - Product review analysis
   - Sentiment extraction
   - Data validation
   - API response parsing

### Embedding & Similarity Features
1. **Document Embedding**
   - Sentence Transformers (all-MiniLM-L6-v2)
   - Vector representation of text
   - Batch document processing

2. **Similarity Search**
   - Cosine similarity computation
   - Query-document matching
   - Ranking by relevance score
   - Real-world ML/AI document examples

## 🎓 Learning Path

### Beginner Level
1. Start with [1.LLMs/1_llm_demo.py](1.LLMs/1_llm_demo.py) - Basic LLM usage
2. Try [2.ChatModels/2_Chatmodel_Antropic.py](2.ChatModels/2_Chatmodel_Antropic.py) - Simple chat model
3. Explore [4_Prompts/messages.py](4_Prompts/messages.py) - Message types

### Intermediate Level
4. Build with [4_Prompts/chatbot.py](4_Prompts/chatbot.py) - Chatbot with memory
5. Work with [3.EmbeeddedModels/2_document_similarity.py](3.EmbeeddedModels/2_document_similarity.py) - Embeddings
6. Try [5_Structured_Output/pydantic_demo.py](5_Structured_Output/pydantic_demo.py) - Data validation

### Advanced Level
7. Create apps with [4_Prompts/prompts.py](4_Prompts/prompts.py) - Streamlit application
8. Extract data with [5_Structured_Output/with_structured_output_pydantic.py](5_Structured_Output/with_structured_output_pydantic.py) - Structured outputs
9. Deploy locally with [2.ChatModels/5_chatmodel_hf_local.py](2.ChatModels/5_chatmodel_hf_local.py) - Local models

## 💡 Key Concepts Demonstrated

### 1. LangChain Core Concepts
- **LLMs vs Chat Models**: Understanding the difference between completion and conversational models
- **Prompt Templates**: Creating reusable, dynamic prompts with variables
- **Message History**: Managing conversation context across multiple turns
- **Embeddings**: Vector representations for semantic search and similarity
- **Structured Output**: Type-safe data extraction from unstructured LLM responses

### 2. Integration Patterns
- **Multi-Provider Support**: Seamlessly switch between OpenAI, Anthropic, Google, and Hugging Face
- **Local vs Cloud**: Deploy models locally (TinyLlama) or use cloud APIs
- **Chain Composition**: Combine templates with models using LCEL (LangChain Expression Language)
- **Streamlit Integration**: Build interactive web applications with LangChain

### 3. Best Practices
- **Environment Variables**: Secure API key management with `.env` files
- **Error Handling**: Rate limiting with time delays between requests
- **Type Safety**: Using Pydantic and TypedDict for data validation
- **Template Persistence**: Saving/loading prompt templates from JSON files

## 🚀 Running Tests

```bash
python test.py
```

## 📖 LangChain Version

This project uses LangChain as specified in `requirement.txt`. Check installed version:
```bash
python -c "import langchain; print(langchain.__version__)"
```

## 🤝 Contributing

Feel free to fork this repository and submit pull requests for improvements.

## 📄 License

This project is open source and available under the MIT License.

## 👨‍💻 Author

**Santosh Chapagain**

GitHub: [github.com/Santosh-Chapagain](https://github.com/Santosh-Chapagain)

## 🔗 Repository

[LangChain_Models](https://github.com/Santosh-Chapagain/LangChain_Models)

## ⚠️ Disclaimer

- Ensure you have appropriate API rate limits set for production use
- Store API keys securely in `.env` files (never commit to repository)
- Review API pricing before running with large datasets
- Respect rate limits of each API provider

## 📞 Support

For issues or questions, please open an issue on the GitHub repository.

---

**Last Updated:** January 2026
