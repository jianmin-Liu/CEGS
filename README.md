# CEGS: Configuration Example Generalizing Synthesizer

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC_BY--NC--SA_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Paper](https://img.shields.io/badge/paper-NSDI%202025-green.svg)](#citation)

CEGS is an advanced network configuration synthesis system that leverages Graph Neural Networks (GNNs) and Large Language Models (LLMs) to automate network configuration synthesis. It can understand high-level user intents, identify and generalize from configuration examples, and generate correct, verifiable network configurations for arbitrary topologies.


## Getting Started

### Prerequisites

- Python 3.8+
- Docker (for Batfish service)
- OpenAI API key (or other supported LLM API)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/cegs.git
   cd cegs/CEGS
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start Batfish service**:
   ```bash
   docker run -d --name batfish -p 9997:9997 -p 9996:9996 batfish/batfish:latest
   ```

4. **Configure environment variables**:
   Set your LLM API key in `setting.json`.

### Configuration

The tool uses a setting file `setting.json` to specify all necessary parameters. You can adjust these parameters according to your needs. Before running, modify the `setting.json` file to set up system parameters, especially the LLM provider and API key.

#### Key Setting Categories:

- **LLM**: Choose from OPENAI, GEMINI, or DEEPSEEK providers and set corresponding API keys and models
- **AI Models**: Set paths for GNN model and sentence transformer  
- **Data Paths**: Set paths for example library, intent types, and input files
- **Output Paths**: Set directories for generated device configurations
- **Prompt Templates**: Set paths for various LLM prompt template files used in different generation phases
- **Generation Parameters**: Set batch size and other synthesis parameters

```json
{
    "LLM_PROVIDER": "OPENAI",
    "GEMINI_API_KEY": "sk-your-gemini-api-key-here",
    "GEMINI_MODEL": "gemini-2.5-flash-preview-05-20",
    "OPENAI_API_KEY": "sk-your-openai-api-key-here", 
    "OPENAI_MODEL": "gpt-4o",
    "DEEPSEEK_API_KEY": "sk-your-deepseek-api-key-here",
    "DEEPSEEK_MODEL": "deepseek-chat",

    "GCN_MODEL_PATH": "best_model.pth",
    "SENTENCE_TRANSFORMER_MODEL": "all-MiniLM-L6-v2",

    "EXAMPLE_LIBRARY_PATH": "ExampleLibrary.json",
    "INTENT_TYPES_PATH": "IntentTypes/Types.json",
    "INPUT_INTENT_FILE": "input/intent.txt",
    "INPUT_TOPOLOGY_FILE": "input/topology.json",

    "RESPONSES_DIR": "responses",
    "OUTPUT_DIR": "output",
    "CONFIGS_DIR": "output/configs/",

    "INTENT_PROCESS_FILE": "prompts/intentprocess.txt",
    "INTENT_SPECIFICATION_FILE": "prompts/intentSpecification.txt",
    "REFLECTION_PREFIX_FILE": "prompts/reflection/prefix.txt",
    "INTERFACE_SUFFIX_FILE": "prompts/reflection/interface_suffix.txt",
    "OSPF_SUFFIX_FILE": "prompts/reflection/ospf_suffix.txt",
    "BGP_SUFFIX_FILE": "prompts/reflection/bgp_suffix.txt",
    "BGP_ROUTING_POLICY_FILE": "prompts/reflection/bgp_routing_policy.txt",
    "ROUTEMAP_FORMAT_FILE": "prompts/reflection/routeMap_format.txt",
    "INTENT_TYPES_ROLE_PROMPT_DIR": "IntentTypes/rolePrompt",

    "BATCH_SIZE": 40
}
```

### Data Dependencies

CEGS relies on pre-trained word embedding models for its semantic understanding capabilities. One of these models is too large to be included directly in this Git repository.

**Required Action:**
Before running the system, you must download the following file and place it in the root directory of the project:

- **File**: `wiki-news-300d-1M-subword.vec`
- **Download URL**: [**Official download link here**](https://fasttext.cc/docs/en/english-vectors.html)
- **Target Location**: `CEGS/wiki-news-300d-1M-subword.vec/wiki-news-300d-1M-subword.vec`

After downloading, your directory structure should look like this:
```
CEGS/
├── wiki-news-300d-1M-subword.vec/
│   └── wiki-news-300d-1M-subword.vec
├── main_syn.py
├── querier.py
└── ... other files
```

### Running the System

To run the synthesis process with a target scenario data as the input, execute the main synthesis script:

```bash
python main_syn.py
```

The system will begin the workflow, processing the intents and topology defined in its input files and generating the final configurations in the `output/` directory.

## Configuration Examples and Data

### Configuration Example Library
The file `ExampleLibrary.json` is a configuration example library containing numerous examples that encompass various routing intents for OSPF and BGP protocols.

### Input Data
The dataset folder contains example target scenario datas. Each data includes intents and target topology.

## Supported Protocols and Intent Types

- **Static Routing**: Static route intents
- **OSPF**: ECMP and Any-path intents
- **BGP**: Simple, Ordered, and No-transit intents


## System Components

### Querier
- Uses two-stage recommendation strategy to identify relevant configuration examples
- Combines semantic similarity and topological similarity for precise matching
- Implemented with FastText and GraphSAGE

### Classifier
- Establishes device associations between target and example topologies
- First performs exact matching based on role descriptions
- Then uses GNN for refinement based on neighborhood similarity

### Verification System
- **Syntax Verifier**: Checks configuration syntax correctness
- **Local Attribute Verifier (LAV)**: Verifies individual device configurations
- **Global Formal Verifier (GFV)**: Uses SMT solver to verify network-wide policies

### Formal Synthesizer
- Based on NetComplete implementation
- Uses SMT constraint solving to fill template parameters
- Guarantees correctness of final configurations


## Project Structure

```
CEGS/
├── main_syn.py              # Main entry point
├── querier.py               # Querier implementation
├── classifier.py            # Classifier implementation
├── generator.py             # Configuration generator
├── utils.py                 # Utility functions
├── Semantic_verifier.py     # Semantic verifier
├── Syntax_verifier.py       # Syntax verifier
├── config_manager.py        # Configuration manager
├── setting.json             # Configuration file
├── prompts/                 # LLM prompt templates
├── dataset/                 # Datasets for training
├── input/                   # Input intent and topology
├── output/                  # Output directory
└── requirements.txt         # Dependencies
```

## License

This project is licensed under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License**.

[![License: CC BY-NC-SA 4.0](https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

This means you are free to:
- **Share** — copy and redistribute the material in any medium or format
- **Adapt** — remix, transform, and build upon the material

Under the following terms:
- **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made.
- **NonCommercial** — You may not use the material for commercial purposes.
- **ShareAlike** — If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original.

## Citation

If you use CEGS in your research, please cite our paper:

```bibtex
@inproceedings {306009,
author = {Jianmin Liu and Li Chen and Dan Li and Yukai Miao},
title = {{CEGS}: Configuration Example Generalizing Synthesizer},
booktitle = {22nd USENIX Symposium on Networked Systems Design and Implementation (NSDI 25)},
year = {2025},
isbn = {978-1-939133-46-5},
address = {Philadelphia, PA},
pages = {1327--1347},
url = {https://www.usenix.org/conference/nsdi25/presentation/liu-jianmin},
publisher = {USENIX Association},
month = apr
}
```


For questions or suggestions, please contact us through GitHub Issues.

---

*This README provides an overview of the CEGS system. For more detailed information on the methodology and evaluation, please refer to the full paper.* 
