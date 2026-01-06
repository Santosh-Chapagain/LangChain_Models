# LangChain Models

A comprehensive project demonstrating various LangChain integrations with multiple Large Language Models (LLMs), Chat Models, and Embedding Models.

## 📋 Project Overview

This project showcases practical implementations of LangChain with different AI providers including:
- **OpenAI (GPT models)**
- **Anthropic (Claude)**
- **Google Generative AI (Gemini)**
- **Hugging Face Models (local and remote)**

## 📁 Project Structure

```
LangChainModel/
├── 1.LLMs/
│   └── 1_llm_demo.py                    # Basic LLM integration with OpenAI
├── 2.ChatModels/
│   ├── 1_Chatmodel_openAI.py           # OpenAI Chat Model implementation
│   ├── 2_Chatmodel_Antropic.py         # Anthropic Claude integration
│   ├── 3_Chatmodel_google.py           # Google Generative AI integration
│   ├── 4_Chatmodel_huggingface.ipynb   # Hugging Face Chat Model
│   └── 5_chatmodel_hf_local.py         # Local Hugging Face model
├── 3.EmbeeddedModels/
│   ├── 1_Embedding_HF_query.py         # Hugging Face embeddings
│   └── 2_document_similarity.py        # Document similarity using embeddings
├── Prompts/
│   └── prompts.py                       # Streamlit research tool with prompt templates
├── requirement.txt                      # Project dependencies
├── test.py                              # Testing script
└── README.md                            # This file
```

## 🚀 Features

### 1. **LLMs (Large Language Models)**
- Interactive question-answering with GPT-3.5 Turbo
- Real-time response handling with 5-second delay between requests

### 2. **Chat Models**
Multiple implementations with different providers:
- **OpenAI**: GPT models for conversational AI
- **Anthropic**: Claude models for advanced reasoning
- **Google Generative AI**: Gemini models for versatile tasks
- **Hugging Face**: Both cloud-based and local model deployments

### 3. **Embedding Models**
- Document embedding generation using Hugging Face
- Document similarity comparison
- Vector-based semantic search capabilities

### 4. **Prompt Engineering**
- **Streamlit Web Application** for research paper explanations
- Dynamic prompt templates with customizable styles:
  - Beginner-Friendly
  - Technical
  - Code-Oriented
  - Mathematical
- Adjustable explanation lengths (Short, Medium, Long)
- Supports multiple research papers:
  - "Attention Is All You Need"
  - "BERT: Pre-training of Deep Bidirectional Transformers"
  - "GPT-3: Language Models are Few-Shot Learners"
  - "Diffusion Models Beat GANs on Image Synthesis"

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
- **streamlit** - Web application framework

## 🎯 Usage Examples

### 1. Running LLM Demo
```bash
python 1.LLMs/1_llm_demo.py
```
Interactive CLI for asking questions to GPT-3.5 Turbo.

### 2. Using Chat Models
```bash
python 2.ChatModels/1_Chatmodel_openAI.py
```

### 3. Running the Streamlit Research Tool
```bash
streamlit run Prompts/prompts.py
```
Access the web interface at `http://localhost:8501`

### 4. Working with Embeddings
```bash
python 3.EmbeeddedModels/1_Embedding_HF_query.py
python 3.EmbeeddedModels/2_document_similarity.py
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

| File | Purpose |
|------|---------|
| `1_llm_demo.py` | Interactive LLM chat interface |
| `1_Chatmodel_openAI.py` | OpenAI chat implementation |
| `2_Chatmodel_Antropic.py` | Claude model integration |
| `3_Chatmodel_google.py` | Google Gemini integration |
| `5_chatmodel_hf_local.py` | Local Hugging Face models |
| `1_Embedding_HF_query.py` | Text embedding generation |
| `2_document_similarity.py` | Semantic document comparison |
| `prompts.py` | Streamlit research tool |

## 🚀 Running Tests

```bash
python test.py
```

## 📖 LangChain Version

This project supports LangChain version indicated in `requirement.txt`. Check installed version:
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
