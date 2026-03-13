# Narrative & Semantic Network Analysis System
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
A comprehensive analysis system for policy documents that combines narrative analysis, structural topic modeling, and semantic network analysis with advanced visualization capabilities.
**叙事分析与语义网络分析系统** - 面向政策文档的综合分析平台，集成叙事分析、结构主题建模、语义网络分析与高级可视化功能。
---
## 📋 Table of Contents
- [Key Features](#-key-features)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Quick Start](#-quick-start)
- [Usage Examples](#-usage-examples)
- [Output Structure](#-output-structure)
- [Advanced Features](#-advanced-features)
- [Visualization Examples](#-visualization-examples)
- [Dependencies](#-dependencies)
---
## 🌟 Key Features
### Core Analysis Modules
| Module | Description |
|--------|-------------|
| **📖 Narrative Analysis** | Analyze narrative arcs, semantic shifts, and discourse patterns across documents |
| **🎯 Structural Topic Modeling** | Topic extraction with NMF/LDA, structural effects analysis |
| **🕸️ Semantic Network Analysis** | Build and visualize semantic networks, export to Gephi format |
| **⚖️ Topic Bias Analysis** | Analyze topic biases across organizations, time periods, and regions |
| **📈 Time Series Analysis** | Track document evolution, emerging/fading tags, and trends |
| **🔄 Topic Evolution** | Topic lifecycle analysis, emerging/fading topic detection |
| **🔍 Document Similarity** | Semantic search and similarity indexing |
| **🌍 Comparative Analysis** | Cross-regional network comparison (e.g., China-US-EU) |
### Advanced Features
- **🔄 Caching System**: Smart embedding and model caching for performance optimization
- **📊 Progress Tracking**: Real-time progress visualization with detailed metrics
- **🎯 Flexible Filtering**: Filter by country, organization type, document type, year range
- **📈 Interactive Visualizations**: Plotly-based interactive HTML charts
- **💾 Gephi Export**: Native GEXF/GraphML support for network visualization
- **🔧 Incremental Learning**: Support for adding new documents to existing models
- **📝 Comprehensive Reports**: Auto-generated analysis reports in multiple formats
---
## 💻 Installation
### Prerequisites
- Python 3.8 or higher
- Required data directories with policy documents
### Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn networkx
pip install sentence-transformers scikit-learn
pip install plotly spacy
pip install python-louvain  # for community detection
```
### Download Spacy Model
```bash
python -m spacy download en_core_web_sm
```
---
## ⚙️ Configuration
Edit the `NarrativeAnalysisConfig` class to configure paths and parameters:
```python
class NarrativeAnalysisConfig:
    def __init__(self):
        # Data paths
        self.metadata_dir = Path("path/to/metadata")
        self.agora_fulltext_dir = Path("path/to/fulltext")
        self.original_data_dir = Path("path/to/DATA")
        
        # Analysis parameters
        self.n_topics = 15  # Number of topics
        self.topic_model_type = 'nmf'  # 'nmf' or 'lda'
        self.min_valid_year = 2015
        self.max_valid_year = 2025
        
        # Filtering options
        self.filter_countries = []  # e.g., ['美国', '中国']
        self.filter_org_types = []  # e.g., ['government']
```
---
## 🚀 Quick Start
### Basic Analysis (All Documents)
```python
python "叙事与语义网络分析系统.py"
```
This will:
1. Load all policy documents
2. Perform comprehensive narrative and semantic analysis
3. Generate interactive visualizations
4. Export networks to Gephi format
5. Generate detailed analysis reports
### Filtered Analysis
```python
# Analyze specific countries
run_analysis_with_country_filter(['美国', '中国'])
# Analyze government documents only
run_analysis_with_org_type_filter(['government'])
```
---
## 📚 Usage Examples
### Example 1: Single Country Analysis
```python
# Analyze only US documents
run_analysis_with_country_filter(['美国'])
```
### Example 2: Multiple Countries
```python
# Compare US and China
run_analysis_with_country_filter(['美国', '中国'])
```
### Example 3: Organization Type Filter
```python
# Analyze government and tech company documents
run_analysis_with_org_type_filter(['government', 'tech_company'])
```
### Example 4: Comparative Analysis (China-US-EU)
```python
# Cross-regional comparative analysis
run_china_us_eu_comparative_analysis()
```
### Example 5: Check Available Filters
```python
# List all available filtering options
list_available_filters()
```
---
## 📁 Output Structure
```
output/
├── narrative_analysis_YYYYMMDD_HHMMSS/
│   ├── analysis_results.json          # Complete analysis results
│   ├── gephi/                         # Network files for Gephi
│   │   ├── semantic_network.gexf
│   │   ├── semantic_network.graphml
│   │   ├── semantic_network_nodes.csv
│   │   └── semantic_network_edges.csv
│   ├── narrative/                     # Narrative analysis
│   │   ├── narrative_arc_heatmap.html
│   │   └── narrative_arc_types.html
│   ├── semantic_network/              # Network visualizations
│   │   ├── semantic_network_40nodes_interactive.html
│   │   ├── semantic_network_80nodes_interactive.html
│   │   └── communities/               # Community sub-networks
│   ├── topic_bias/                    # Topic bias analysis
│   │   ├── topic_bias_heatmap.html
│   │   └── topic_trends.html
│   ├── time_series/                   # Time series analysis
│   │   ├── document_timeline.html
│   │   ├── tag_evolution_heatmap.html
│   │   └── topic_evolution.html
│   ├── reports/                       # Analysis reports
│   │   ├── analysis_report.txt
│   │   └── topic_top20_words.txt
│   └── comparative_analysis/          # Comparative analysis (if applicable)
│       ├── comparative_analysis_report.txt
│       ├── topology_radar.html
│       └── network_similarity_matrix.html
.cache/                                # Cache directory
├── embeddings/                        # Cached embeddings
└── models/                            # Cached models
```
---
## 🔬 Advanced Features
### Caching System
The system automatically caches:
- **Embeddings**: Document and segment embeddings
- **Models**: Trained topic models
- **Results**: Analysis results
View cache statistics:
```python
cache_stats = cache_manager.get_cache_stats()
# Hit rate, cache size, etc.
```
### Incremental Learning
Add new documents to existing models:
```python
incremental_manager = IncrementalLearningManager(config, cache_manager)
incremental_manager.initialize_with_model(topic_model, vectorizer, documents)
incremental_manager.add_documents(new_documents, topic_analyzer)
```
### Document Similarity Search
```python
# Build similarity index
doc_search = DocumentSimilaritySearch(config, cache_manager)
doc_search.build_index(documents)
# Search similar documents
similar_docs = doc_search.search_similar("query text", top_k=10)
```
---
## 📊 Visualization Examples
### 1. Narrative Arc Heatmap
Interactive heatmap showing semantic shifts across document segments.
### 2. Semantic Network Visualization
Multi-scale interactive network graphs with community detection.
### 3. Topic Evolution Timeline
Track how topics change over time with trend indicators.
### 4. Comparative Analysis Dashboard
Radar charts, similarity matrices, and core concept comparisons.
---
## 📋 Analysis Stages
The system executes 13 comprehensive stages:
| Stage | Description |
|-------|-------------|
| 1️⃣ | Initialize Configuration |
| 2️⃣ | Load Stopwords |
| 3️⃣ | Load Metadata |
| 4️⃣ | Load & Filter Documents |
| 5️⃣ | Narrative Analysis |
| 6️⃣ | Structural Topic Analysis |
| 7️⃣ | Semantic Network Analysis |
| 8️⃣ | Topic Bias Analysis |
| 9️⃣ | Time Series Analysis |
| 🔟 | Topic Evolution Analysis |
| 1️⃣1️⃣ | Document Similarity Indexing |
| 1️⃣2️⃣ | Save Results |
| 1️⃣3️⃣ | Generate Reports |
---
## 🛠️ Dependencies
### Core Libraries
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `networkx` - Network analysis
- `scikit-learn` - Machine learning (NMF, LDA, TF-IDF)
- `sentence-transformers` - Text embeddings
### Visualization
- `plotly` - Interactive visualizations
- `matplotlib` - Static plots
- `seaborn` - Statistical visualizations
### NLP
- `spacy` - Text processing
- `python-louvain` (community) - Community detection
### Optional
- `tqdm` - Progress bars
---
## 📈 Performance Optimization
- **Embedding Caching**: Avoid recomputing embeddings for same text
- **Batch Processing**: Process documents in batches for memory efficiency
- **Lazy Loading**: Load models only when needed
- **Progress Tracking**: Silent progress iterator to reduce I/O overhead
---
## 🔍 Filtering Options
### Available Countries
```
美国, 中国, 英国, 欧盟, 日本, 加拿大, 澳大利亚, 国际组织, etc.
```
### Available Organization Types
```
government, tech_company, international_org, academia, civil_society, etc.
```
### Available Document Types
```
national_guideline, white_paper, legislation, report, policy, etc.
```
---
## 📝 Output File Formats
| Type | Formats |
|------|---------|
| **Network Files** | GEXF, GraphML, CSV (nodes/edges) |
| **Visualizations** | HTML (Plotly interactive) |
| **Reports** | TXT, JSON, CSV |
| **Topic Wordlists** | TXT, CSV (detailed) |
| **Statistics** | JSON, TXT |
---
## 🤝 Contributing
Contributions are welcome! Please feel free to submit issues or pull requests.
---
## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
---
## 📧 Contact
For questions or support, please open an issue on GitHub.
---
## 🙏 Acknowledgments
- Sentence Transformers for embedding models
- NetworkX for network analysis
- Plotly for interactive visualizations
- Gephi for network visualization platform
---
**Happy Analyzing! 🎉**
