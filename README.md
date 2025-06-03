# GitHub Issue Searcher MCP 🚀

A powerful **Model Context Protocol (MCP) Server** that combines **FAISS** (Facebook AI Similarity Search) with **OpenAI embeddings** to provide semantic search capabilities for GitHub repository issues. Perfect for building RAG (Retrieval-Augmented Generation) systems and knowledge bases focused on GitHub issue tracking!

## ✨ Features

- 🔍 **Semantic Search**: Generate embeddings using OpenAI and search with FAISS
- 📚 **Document Storage**: Store text with rich metadata for organized retrieval  
- 🐙 **GitHub Integration**: Automatically sync and search GitHub repository issues
- ⚡ **Fast Similarity Search**: Powered by Facebook's FAISS library
- 🔧 **MCP Compatible**: Works seamlessly with Claude and other MCP clients
- 💾 **Persistent Storage**: Automatic saving and loading of indexes and metadata

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- OpenAI API key
- (Optional) GitHub Personal Access Token for GitHub integration

### Quick Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/calumjs/github-issue-searcher-mcp.git
   cd github-issue-searcher-mcp
   ```

2. **Set up virtual environment (recommended)**
   ```bash
   cd faiss_mcp_server
   python -m venv .venv
   
   # Activate virtual environment
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the `faiss_mcp_server` directory:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   GITHUB_PERSONAL_ACCESS_TOKEN=your_github_token_here  # Optional
   ```

5. **Sync a GitHub repository (Essential First Step)**
   ```bash
   # Sync your first repository to populate the search index
   python sync_github.py owner/repo --clear
   
   # Examples:
   python sync_github.py microsoft/vscode --clear
   python sync_github.py facebook/react --clear
   python sync_github.py SSWConsulting/SSW.YakShaver --clear
   ```

6. **Configure your MCP client**
   
   Add the server to your MCP client configuration. For **Claude Desktop**, add this to your MCP settings:
   
   ```json
   {
     "mcpServers": {
       "github-issue-searcher": {
         "command": "python",
         "args": ["/path/to/your/faiss_mcp_server/main.py"],
         "env": {
           "OPENAI_API_KEY": "your-openai-api-key-here"
         }
       }
     }
   }
   ```
   
   **Note**: Replace `/path/to/your/faiss_mcp_server/main.py` with the actual path to your installation.
   
   **Alternative**: You can also run the server directly for testing:
   ```bash
   python main.py
   ```

## 🚀 Usage

### Step 1: Sync GitHub Repository Data

**This is the essential first step!** The GitHub Issue Searcher MCP needs repository data to search through. You have two options:

**Option 1: Standalone sync script (Recommended)**
```bash
cd faiss_mcp_server

# Sync a single repository (clears existing data)
python sync_github.py owner/repo --clear

# Add multiple repositories (append to existing data)
python sync_github.py microsoft/vscode
python sync_github.py facebook/react
python sync_github.py vercel/next.js
```

**Option 2: Sync during server startup**
```bash
cd faiss_mcp_server

# Start server and sync repository
python main.py --sync-repo owner/repo --clear
```

**Important Notes:**
- Use `--clear` flag to replace existing data with new repository
- Omit `--clear` flag to add repository data to existing index
- Syncing can take several minutes depending on repository size
- The server remains responsive during standalone sync operations

### Step 2: Search Through Issues

Once you have synced repository data, you can use the MCP tools:

#### **search_embeddings** 
Search for similar content using natural language:
```python
search_embeddings(
    query_text="How to fix memory leaks in React components?",
    k=5  # Number of results to return
)
```

#### **get_index_status**
Check what data is currently indexed:
```python
get_index_status()
# Returns: {"initialized": true, "dimension": 1536, "vector_count": 4490, ...}
```

### Managing Your Data

**Add More Repositories:**
```bash
# Add additional repositories without clearing existing data
python sync_github.py tensorflow/tensorflow
python sync_github.py pytorch/pytorch
```

**Replace All Data:**
```bash
# Clear everything and sync a new repository
python sync_github.py new-owner/new-repo --clear
```

**Check Current Data:**
Use the `get_index_status` tool to see how many issues are currently indexed and from which repositories.

## 🏗️ Architecture

```
faiss_mcp_server/
├── main.py              # Entry point and argument parsing
├── sync_github.py       # Standalone GitHub sync script
├── server.py            # MCP server initialization
├── requirements.txt     # Python dependencies
├── pyproject.toml       # Project configuration
├── tools/
│   ├── faiss_tools.py   # FAISS search tools
│   └── github_tools.py  # GitHub integration utilities
└── utils/
    ├── faiss_handler.py # Core FAISS operations
    └── github_syncer.py # GitHub API integration
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | Your OpenAI API key for embeddings | ✅ Yes |
| `GITHUB_PERSONAL_ACCESS_TOKEN` | GitHub token for repository access | ❌ Optional |

### Data Storage

- **FAISS Index**: Stored as `faiss_index.bin`
- **Metadata**: Stored as `metadata.json`
- **Location**: `data/` directory (created automatically)

## 📊 Use Cases

- **🐛 GitHub Issue Search**: Semantically search through GitHub repository issues
- **📖 Issue Knowledge Base**: Build searchable knowledge bases from GitHub issues
- **🤖 RAG Systems**: Enhance AI responses with relevant GitHub issue context
- **📚 Research**: Find similar issues, bug reports, or feature requests
- **💬 Developer Support**: Search through issue history for troubleshooting
- **🔍 Project Management**: Discover related issues and track patterns

## 🛡️ Error Handling

The server includes comprehensive error handling:
- Automatic retry for API failures
- Graceful degradation when services are unavailable  
- Detailed logging for debugging
- Input validation and sanitization

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests if applicable
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Facebook Research** for the amazing FAISS library
- **OpenAI** for the embedding models
- **Anthropic** for the Model Context Protocol
- **GitHub** for the excellent API

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/calumjs/github-issue-searcher-mcp/issues) page
2. Create a new issue with detailed information
3. Include logs and error messages when possible

---

**Made with ❤️ for the GitHub community**

*Happy issue searching! 🔍✨* 