# LangChain Models

A comprehensive project demonstrating various LangChain integrations with multiple Large Language Models (LLMs), Chat Models, Embedding Models, Prompt Engineering, Structured Output handling, Chains, Runnables, and complete RAG (Retrieval-Augmented Generation) implementations.

## 📋 Project Overview

This project showcases practical implementations of LangChain with different AI providers and advanced LangChain patterns including:
- **OpenAI (GPT models)**
- **Anthropic (Claude)**
- **Google Generative AI (Gemini)**
- **Hugging Face Models (local and remote)**
- **Advanced Prompt Engineering**
- **Structured Output with Pydantic and TypedDict**
- **Output Parsers for Open Source Models**
- **LangChain Chains (Simple, Sequential, Parallel, Conditional)**
- **Runnables (Sequence, Parallel, Passthrough, Lambda, Branch)**
- **Complete RAG Pipeline (Document Loaders, Text Splitters, Vector Stores, Retrievers)**

## 📁 Project Structure

```
LangChainModel/
├── 1.LLMs/
│   └── 1_llm_demo.py                           # Basic LLM integration with OpenAI
├── 2.ChatModels/
│   ├── 1_Chatmodel_openAI.py                   # OpenAI Chat Model implementation
│   ├── 2_Chatmodel_Antropic.py                 # Anthropic Claude integration
│   ├── 3_Chatmodel_google.py                   # Google Generative AI integration
│   ├── 4_Chatmodel_huggingface.ipy             # Hugging Face Chat Model
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
├── 6_OutputParser_Opensourcemodel/
│   ├── jsonoutput_parser.py                    # JSON output parsing
│   ├── pydantic_structured_output.py.py        # Pydantic output parser
│   ├── string_outputparser.py                  # Basic string output parser
│   └── stringoutputparser1.py                  # Advanced string output parsing
├── 7_Chains/
│   ├── 1_simple_chain.py                       # Simple chain with prompt | model | parser
│   ├── 2_sequential_chain.py                   # Sequential chain processing
│   ├── 3_Parallel_Chain.py                     # Parallel chain execution
│   └── 4_Conditional_Chains.py                 # Conditional branching chains
├── 8_Runnables/
│   ├── 1_Runnable_Sequence.py                  # Sequential runnable composition
│   ├── 2_Runnable_Parallel.py                  # Parallel runnable execution
│   ├── 3_Runnable-Passthrough.py               # Passthrough data in runnables
│   ├── 4_Runnable_Lambda.py                    # Custom lambda functions
│   └── 5_Runnable_Branch.py                    # Conditional runnable branching
├── 9_RAG/
│   ├── 1_DocumentLoader/
│   │   ├── 1_text_loader.py                    # Load text files
│   │   ├── 2_PyPDFLoader.py                    # Load and process PDF documents
│   │   ├── 3_Directory_Loader.py               # Load entire directories
│   │   ├── 4_Web_Based_Loader.py               # Scrape and load web content
│   │   └── poem.txt                            # Sample text file
│   ├── 2_Text_Splitter/
│   │   ├── 1_Charactertextsplitter.py          # Split by character count
│   │   ├── 2_Recursive_text_splitter.py        # Recursive text splitting
│   │   └── 3_Python_Code_Splitting.py          # Code-aware text splitting
│   ├── 3_Vector_Store/
│   │   ├── 1_Chroma_db.ipynb                   # Chroma vector database
│   │   └── chroma_db/                          # Persisted Chroma database
│   └── 4_Retrievers/
│       ├── 1_Wikipedia_Retriever.ipynb         # Wikipedia knowledge retrieval
│       ├── 2_Vector_Store_Retriever.ipynb      # Vector store-based retrieval
│       ├── 3_MMR.ipynb                         # Maximum Marginal Relevance
│       ├── 4_Multiquery_Retriever.ipynb        # Multi-query retrieval strategy
│       └── 5_Contexual_Compression_Retriever.ipynb  # Context compression
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
- **Hugging Face**: Both cloud-based and local model deployments (TinyLlama, Mistral)

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

### 6. **Output Parsers for Open Source Models**
- **JSON Output Parser**: Parse JSON responses with format instructions
- **String Output Parser**: Basic string parsing and cleaning
- **Pydantic Output Parser**: Type-safe structured output for open source models
- **Format Instructions**: Auto-generate prompts for structured output from Hugging Face models

### 7. **LangChain Chains**
- **Simple Chain**: Basic prompt → model → parser pattern using LCEL (|)
- **Sequential Chain**: Multi-step processing (generate joke → explain joke)
- **Parallel Chain**: Execute multiple chains simultaneously and merge results
- **Conditional Chain**: Route to different chains based on sentiment classification
- **Graph Visualization**: Display chain execution flow with `.get_graph().print_ascii()`

### 8. **Runnables**
- **RunnableSequence**: Explicitly define sequential processing steps
- **RunnableParallel**: Execute multiple operations in parallel with dictionary output
- **RunnablePassthrough**: Pass input data through unchanged to next step
- **RunnableLambda**: Custom Python functions in chains
- **RunnableBranch**: Conditional routing based on lambda predicates

### 9. **RAG (Retrieval-Augmented Generation)**

#### 9.1 Document Loaders
- **TextLoader**: Load plain text files with encoding support
- **PyPDFLoader**: Extract text from PDF documents page by page
- **DirectoryLoader**: Batch load multiple files from directories
- **WebBaseLoader**: Scrape and load content from web URLs

#### 9.2 Text Splitters
- **CharacterTextSplitter**: Split by character count with customizable separators
- **RecursiveCharacterTextSplitter**: Smart splitting with chunk overlap
- **Python Code Splitter**: Code-aware splitting preserving structure

#### 9.3 Vector Stores
- **Chroma DB**: Persistent vector database with local storage
- **FAISS**: Facebook AI Similarity Search for efficient vector operations
- **Embedding Integration**: Use Hugging Face embeddings for vector creation

#### 9.4 Retrievers
- **Wikipedia Retriever**: Fetch information directly from Wikipedia
- **Vector Store Retriever**: Query similarity-based document retrieval
- **MMR (Maximum Marginal Relevance)**: Diverse result retrieval avoiding redundancy
- **Multi-Query Retriever**: Generate multiple query variations for comprehensive search
- **Contextual Compression Retriever**: Extract relevant portions using LLM compression

## 📦 Installation

### Prerequisites
- Python 3.8+
- Virtual Environment (recommended)

### Setup Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Santosh-Chapagain/LangChain.git
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
     HUGGINGFACEHUB_ACCESS_TOKEN=your_huggingface_token
     ```

## 📚 Dependencies

- **langchain** - Core LangChain framework
- **langchain-core** - Core abstractions
- **langchain-openai** - OpenAI integration
- **langchain-anthropic** - Anthropic integration
- **langchain-google-genai** - Google Generative AI integration
- **langchain-huggingface** - Hugging Face integration
- **langchain-community** - Community integrations (document loaders, retrievers)
- **langchain-text-splitters** - Text splitting utilities
- **transformers** - Transformers library for HF models
- **huggingface-hub** - Hugging Face Hub access
- **openai** - OpenAI client library
- **google-generativeai** - Google Generative AI client
- **python-dotenv** - Environment variable management
- **pydantic** - Data validation and settings management
- **chromadb** - Chroma vector database
- **faiss-cpu** - Facebook AI Similarity Search
- **pypdf** - PDF processing
- **beautifulsoup4** - Web scraping for WebBaseLoader
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

### 6. Output Parser Examples

**JSON Output Parser:**
```bash
python 6_OutputParser_Opensourcemodel/jsonoutput_parser.py
```

**String Output Parser:**
```bash
python 6_OutputParser_Opensourcemodel/string_outputparser.py
```

**Pydantic Structured Output:**
```bash
python 6_OutputParser_Opensourcemodel/pydantic_structured_output.py.py
```

### 7. Chain Examples

**Simple Chain:**
```bash
python 7_Chains/1_simple_chain.py
```

**Sequential Chain:**
```bash
python 7_Chains/2_sequential_chain.py
```

**Parallel Chain:**
```bash
python 7_Chains/3_Parallel_Chain.py
```

**Conditional Chain:**
```bash
python 7_Chains/4_Conditional_Chains.py
```

### 8. Runnable Examples

**Runnable Sequence:**
```bash
python 8_Runnables/1_Runnable_Sequence.py
```

**Runnable Parallel:**
```bash
python 8_Runnables/2_Runnable_Parallel.py
```

**Runnable Passthrough:**
```bash
python 8_Runnables/3_Runnable-Passthrough.py
```

**Runnable Lambda:**
```bash
python 8_Runnables/4_Runnable_Lambda.py
```

**Runnable Branch:**
```bash
python 8_Runnables/5_Runnable_Branch.py
```

### 9. RAG Examples

**Document Loaders:**
```bash
python 9_RAG/1_DocumentLoader/1_text_loader.py
python 9_RAG/1_DocumentLoader/2_PyPDFLoader.py
python 9_RAG/1_DocumentLoader/4_Web_Based_Loader.py
```

**Text Splitters:**
```bash
python 9_RAG/2_Text_Splitter/1_Charactertextsplitter.py
python 9_RAG/2_Text_Splitter/2_Recursive_text_splitter.py
```

**Vector Store (Jupyter Notebook):**
```bash
jupyter notebook 9_RAG/3_Vector_Store/1_Chroma_db.ipynb
```

**Retrievers (Jupyter Notebooks):**
```bash
jupyter notebook 9_RAG/4_Retrievers/1_Wikipedia_Retriever.ipynb
jupyter notebook 9_RAG/4_Retrievers/2_Vector_Store_Retriever.ipynb
jupyter notebook 9_RAG/4_Retrievers/3_MMR.ipynb
jupyter notebook 9_RAG/4_Retrievers/4_Multiquery_Retriever.ipynb
jupyter notebook 9_RAG/4_Retrievers/5_Contexual_Compression_Retriever.ipynb
```

## 🔑 API Keys Required

| Service | File | Environment Variable |
|---------|------|----------------------|
| OpenAI | `.env` | `OPENAI_API_KEY` |
| Anthropic | `.env` | `ANTHROPIC_API_KEY` |
| Google | `.env` | `GOOGLE_API_KEY` |
| Hugging Face | `.env` | `HUGGINGFACEHUB_ACCESS_TOKEN` |

## 🛠️ Configuration

All configurations are managed through:
- `.env` file for sensitive keys
- Streamlit app settings in `4_Prompts/prompts.py`
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
| `4_Chatmodel_huggingface.ipy` | Jupyter notebook for HF models | Notebook-based experimentation |
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

### 6_OutputParser_Opensourcemodel/
| File | Purpose | Key Features |
|------|---------|--------------|
| `jsonoutput_parser.py` | JSON parsing for open source | Parse JSON with format instructions for Mistral/Llama models |
| `string_outputparser.py` | Basic string parsing | Clean and parse string outputs |
| `stringoutputparser1.py` | Advanced string parsing | Enhanced string output handling |
| `pydantic_structured_output.py.py` | Pydantic for open source models | Type-safe structured output from Hugging Face models |

### 7_Chains/
| File | Purpose | Key Features |
|------|---------|--------------|
| `1_simple_chain.py` | Basic LCEL chain | Prompt → Model → Parser using pipe operator |
| `2_sequential_chain.py` | Multi-step processing | Summary generation → Key points extraction |
| `3_Parallel_Chain.py` | Concurrent execution | Generate notes and quiz simultaneously, then merge |
| `4_Conditional_Chains.py` | Sentiment-based routing | Route feedback to positive/negative response chains |

### 8_Runnables/
| File | Purpose | Key Features |
|------|---------|--------------|
| `1_Runnable_Sequence.py` | Explicit sequential composition | Generate joke → Explain joke pipeline |
| `2_Runnable_Parallel.py` | Parallel execution with dict output | Simultaneous operations with named outputs |
| `3_Runnable-Passthrough.py` | Data passthrough | Preserve input while processing |
| `4_Runnable_Lambda.py` | Custom functions in chains | Integrate Python functions into runnables |
| `5_Runnable_Branch.py` | Conditional branching | Route based on output length (summarize if > 300 words) |

### 9_RAG/

#### 1_DocumentLoader/
| File | Purpose | Key Features |
|------|---------|--------------|
| `1_text_loader.py` | Load text files | Read poem.txt with encoding support |
| `2_PyPDFLoader.py` | PDF document processing | Extract text from PDF, generate Q&A with Mistral |
| `3_Directory_Loader.py` | Batch file loading | Load multiple files from directories |
| `4_Web_Based_Loader.py` | Web content scraping | Fetch and parse web pages |
| `poem.txt` | Sample text file | Test data for document loaders |

#### 2_Text_Splitter/
| File | Purpose | Key Features |
|------|---------|--------------|
| `1_Charactertextsplitter.py` | Character-based splitting | Split PDF into 100-character chunks |
| `2_Recursive_text_splitter.py` | Smart recursive splitting | 300-character chunks with 10-character overlap |
| `3_Python_Code_Splitting.py` | Code-aware splitting | Preserve code structure during splitting |

#### 3_Vector_Store/
| File | Purpose | Key Features |
|------|---------|--------------|
| `1_Chroma_db.ipynb` | Chroma vector database | Persistent vector storage with Hugging Face embeddings |
| `chroma_db/` | Database files | SQLite database and vector indices |

#### 4_Retrievers/
| File | Purpose | Key Features |
|------|---------|--------------|
| `1_Wikipedia_Retriever.ipynb` | Wikipedia API integration | Fetch supermassive black hole information |
| `2_Vector_Store_Retriever.ipynb` | Similarity search | Query Chroma DB for relevant documents |
| `3_MMR.ipynb` | Maximum Marginal Relevance | Diverse results using FAISS |
| `4_Multiquery_Retriever.ipynb` | Multi-query retrieval | Generate multiple query variations |
| `5_Contexual_Compression_Retriever.ipynb` | Context compression | Extract relevant portions using LLM |

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

### Chain Patterns
1. **LCEL (LangChain Expression Language)**
   - Pipe operator (|) for chain composition
   - Type-safe chain building
   - Automatic graph generation with `.get_graph().print_ascii()`

2. **Chain Types**
   - **Simple**: prompt | model | parser
   - **Sequential**: output of one chain → input of next
   - **Parallel**: multiple chains execute simultaneously
   - **Conditional**: route based on classification or predicates

3. **Use Cases**
   - Document summarization with key points
   - Note generation and quiz creation in parallel
   - Sentiment-based customer feedback routing
   - Joke generation with explanations

### Runnable Patterns
1. **Core Runnables**
   - **RunnableSequence**: Explicit pipeline definition
   - **RunnableParallel**: Named parallel outputs in dictionary
   - **RunnablePassthrough**: Pass data unchanged to next step
   - **RunnableLambda**: Custom Python functions in chains
   - **RunnableBranch**: Conditional routing with lambda predicates

2. **Advanced Patterns**
   - Preserve original input while processing
   - Conditional summarization based on length
   - Merge parallel outputs (jokes + explanations)
   - Dynamic routing based on runtime conditions

### RAG Pipeline
1. **Document Loading**
   - Multi-format support (TXT, PDF, Web)
   - Encoding handling (UTF-8)
   - Batch processing with DirectoryLoader
   - Web content extraction with BeautifulSoup

2. **Text Splitting Strategies**
   - Character-based: Fixed chunk size
   - Recursive: Smart splitting with overlap
   - Code-aware: Preserve structure for Python code
   - Customizable separators and chunk overlap

3. **Vector Storage**
   - **Chroma**: Persistent local database with SQLite backend
   - **FAISS**: High-performance similarity search from Facebook AI
   - Embedding integration with Hugging Face models
   - Document metadata preservation

4. **Retrieval Strategies**
   - **Similarity Search**: Find most relevant documents by vector distance
   - **MMR**: Balance relevance and diversity in results
   - **Multi-Query**: Expand single query into multiple variations
   - **Contextual Compression**: Extract only relevant portions with LLM
   - **Wikipedia**: Direct knowledge base access

## 🎓 Learning Path

### Beginner Level (Fundamentals)
1. Start with [1.LLMs/1_llm_demo.py](1.LLMs/1_llm_demo.py) - Basic LLM usage
2. Try [2.ChatModels/2_Chatmodel_Antropic.py](2.ChatModels/2_Chatmodel_Antropic.py) - Simple chat model
3. Explore [4_Prompts/messages.py](4_Prompts/messages.py) - Message types
4. Learn [5_Structured_Output/pydantic_demo.py](5_Structured_Output/pydantic_demo.py) - Data validation

### Intermediate Level (Integration)
5. Build with [4_Prompts/chatbot.py](4_Prompts/chatbot.py) - Chatbot with memory
6. Work with [3.EmbeeddedModels/2_document_similarity.py](3.EmbeeddedModels/2_document_similarity.py) - Embeddings
7. Practice [7_Chains/1_simple_chain.py](7_Chains/1_simple_chain.py) - Chain basics
8. Explore [6_OutputParser_Opensourcemodel/jsonoutput_parser.py](6_OutputParser_Opensourcemodel/jsonoutput_parser.py) - Output parsing

### Advanced Level (Complex Patterns)
9. Master [7_Chains/3_Parallel_Chain.py](7_Chains/3_Parallel_Chain.py) - Parallel execution
10. Learn [8_Runnables/5_Runnable_Branch.py](8_Runnables/5_Runnable_Branch.py) - Conditional logic
11. Build [9_RAG/1_DocumentLoader/2_PyPDFLoader.py](9_RAG/1_DocumentLoader/2_PyPDFLoader.py) - Document processing
12. Implement [9_RAG/4_Retrievers/3_MMR.ipynb](9_RAG/4_Retrievers/3_MMR.ipynb) - Advanced retrieval

### Expert Level (Production Systems)
13. Create apps with [4_Prompts/prompts.py](4_Prompts/prompts.py) - Streamlit application
14. Extract data with [5_Structured_Output/with_structured_output_pydantic.py](5_Structured_Output/with_structured_output_pydantic.py) - Structured outputs
15. Deploy locally with [2.ChatModels/5_chatmodel_hf_local.py](2.ChatModels/5_chatmodel_hf_local.py) - Local models
16. Build RAG with [9_RAG/4_Retrievers/5_Contexual_Compression_Retriever.ipynb](9_RAG/4_Retrievers/5_Contexual_Compression_Retriever.ipynb) - Complete system

## 💡 Key Concepts Demonstrated

### 1. LangChain Core Concepts
- **LLMs vs Chat Models**: Understanding the difference between completion and conversational models
- **Prompt Templates**: Creating reusable, dynamic prompts with variables
- **Message History**: Managing conversation context across multiple turns
- **Embeddings**: Vector representations for semantic search and similarity
- **Structured Output**: Type-safe data extraction from unstructured LLM responses
- **Chains**: Composing multi-step workflows with LCEL
- **Runnables**: Building flexible, composable processing pipelines
- **RAG**: Complete retrieval-augmented generation workflows

### 2. Integration Patterns
- **Multi-Provider Support**: Seamlessly switch between OpenAI, Anthropic, Google, and Hugging Face
- **Local vs Cloud**: Deploy models locally (TinyLlama) or use cloud APIs
- **Chain Composition**: Combine templates with models using LCEL (LangChain Expression Language)
- **Parallel Processing**: Execute multiple chains simultaneously for efficiency
- **Conditional Routing**: Dynamic chain selection based on runtime conditions
- **Vector Databases**: Persistent storage with Chroma and FAISS
- **Streamlit Integration**: Build interactive web applications with LangChain

### 3. Best Practices
- **Environment Variables**: Secure API key management with `.env` files
- **Error Handling**: Rate limiting with time delays between requests
- **Type Safety**: Using Pydantic and TypedDict for data validation
- **Template Persistence**: Saving/loading prompt templates from JSON files
- **Chunk Overlap**: Smart text splitting for context preservation
- **Document Metadata**: Tracking source information in RAG pipelines
- **Graph Visualization**: Debug chains with `.get_graph().print_ascii()`

### 4. Advanced Techniques
- **MMR (Maximum Marginal Relevance)**: Balance relevance and diversity
- **Multi-Query Retrieval**: Generate query variations for comprehensive search
- **Contextual Compression**: Extract relevant portions with LLM filtering
- **Format Instructions**: Auto-generate prompts for structured output
- **Sentiment Routing**: Classify and route to appropriate response chains
- **Parallel Chain Merging**: Combine outputs from simultaneous operations

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

[LangChain](https://github.com/Santosh-Chapagain/LangChain)

## ⚠️ Disclaimer

- Ensure you have appropriate API rate limits set for production use
- Store API keys securely in `.env` files (never commit to repository)
- Review API pricing before running with large datasets
- Respect rate limits of each API provider
- Hugging Face API tokens required for cloud inference
- Some retrievers (Wikipedia, Web) require internet connection
- Vector databases (Chroma, FAISS) persist locally - ensure sufficient disk space

## 📞 Support

For issues or questions, please open an issue on the GitHub repository.

---

**Last Updated:** January 2026
