# CEGS: Configuration Example Generalizing Synthesizer

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC_BY--NC--SA_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Paper](https://img.shields.io/badge/paper-NSDI%202025-green.svg)](#citation)

This repository contains the official implementation of **CEGS: Configuration Example Generalizing Synthesizer**.

CEGS is an advanced network configuration synthesis system that leverages Graph Neural Networks (GNNs) and Large Language Models (LLMs) to automate network configuration synthesis. It can understand high-level user intents, identify and generalize from configuration examples, and generate correct, verifiable network configurations for arbitrary topologies.

## Abstract

> Network configuration synthesis promises to increase the efficiency of network management by reducing human involvement. However, despite significant advances in this field, existing synthesizers still require much human effort in drafting configuration templates or coding in a domain-specific language. We argue that the main reason for this is that a core capability is missing for current synthesizers: identifying and following configuration examples in configuration manuals and generalizing them to arbitrary topologies.
>
> In this work, we fill this capability gap with two recent advancements in artificial intelligence: graph neural networks (GNNs) and large language models (LLMs). We build CEGS, which can automatically identify appropriate configuration examples, follow and generalize them to fit target network scenarios. CEGS features a GNN-based Querier to identify relevant examples from device documentations, a GNN-based Classifier to generalize the example to arbitrary topology, and an efficient LLM-driven synthesis method to quickly and correctly synthesize configurations that comply with the intents. Evaluations of real-world networks and complex intents show that CEGS can automatically synthesize correct configurations for a network of 1094 devices without human involvement. In contrast, the state-of-the-art LLM-based synthesizer are more than 30 times slower than CEGS on average, even when human experts are in the loop.

## System Workflow

CEGS operates in three main phases to translate high-level user intents into device-specific configurations:

### 1. Retrieval Phase
The GNN-based **Querier** identifies the most relevant configuration example from a knowledge base (i.e., device documentation). It does this by analyzing both the semantic similarity of the user's intent and the topological similarity of the associated network graph.

### 2. Association Phase
The GNN-based **Classifier** establishes a precise mapping between the devices in the target topology and the nodes in the retrieved example's topology. This is crucial for correctly applying configuration snippets from the example to the new scenario.

### 3. Generation Phase
An iterative, LLM-driven synthesis method generates the final configuration:
- **Template Generation**: The LLM generates configuration *templates* for devices, leaving policy-specific parameters (like IP addresses or BGP community values) as symbolic placeholders.
- **Verification (LAV & GFV)**: The templates are rigorously checked for correctness. The **Local Attribute Verifier (LAV)** checks device-specific settings, while the **Global Formal Verifier (GFV)** uses an SMT solver to ensure the network-wide logic fulfills the user's intent.
- **Iterative Correction**: If any errors are found, the verifiers provide structured feedback to the LLM, which then corrects the templates in the next iteration.
- **Formal Synthesis**: Once the templates are verified, a **Formal Synthesizer** (based on NetComplete) calculates the concrete values for all symbolic parameters, producing the final, correct, network-wide configuration.

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

#### Key Configuration Categories:

- **LLM Configuration**: Choose from OPENAI, GEMINI, or DEEPSEEK providers and set corresponding API keys and models
- **AI Models**: Configure paths for GCN model and sentence transformer for semantic understanding  
- **Data Paths**: Set paths for example library, intent types, and input files
- **Output Paths**: Configure directories for responses, output configurations, and generated configs
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
- **Download URL**: [**Please add the official download link here**](https://fasttext.cc/docs/en/english-vectors.html)
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

### Routing Protocols
- **Static Routing**: Static route intents
- **OSPF**: ECMP and Any-path intents
- **BGP**: Simple, Ordered, and No-transit intents

### Intent Examples
```
# BGP path preference intent
"For traffic from Miami to other networks, prefer path (Miami, FortLa, Hollyw, Miami) over path (Miami, FortLa, NodeID85, Orland, Tampa, Miami)"

# OSPF ECMP intent
"Traffic from source to destination is load-balanced between path1 and path2"

# BGP No-transit intent
"Prohibit AS1 and AS2 from forwarding transit traffic through the current network"
```

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

## Performance Evaluation

### Comparison with Existing Systems
| System | Auto Loops | Synthesis Time | Human Intervention |
|--------|------------|----------------|-------------------|
| CEGS | 2-10 | 24s-24m | Not required |
| COSYNTH | 300 (Failed) | >1 hour | Required |
| NETBUDDY | 300 (Failed) | >3 hours | Required |

### Scalability Performance
- **Small Networks** (20-50 devices): 42s-3m25s
- **Medium Networks** (150-200 devices): 1m30s-6m55s
- **Large Networks** (1094 devices): 5m35s-24m50s

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
@inproceedings{liu2024cegs,
  title={{CEGS}: {Configuration} {Example} {Generalizing} {Synthesizer}},
  author={Liu, Jianmin and Chen, Li and Li, Dan and Miao, Yukai},
  booktitle={21st USENIX Symposium on Networked Systems Design and Implementation (NSDI 24)},
  pages={XXX--XXX},
  year={2024},
  organization={USENIX Association}
}
```

## Limitations

As discussed in the paper, CEGS has the following limitations:

- **Documentation Coverage**: The system's performance is constrained by the quality and coverage of the provided device documentation. If a relevant example for a user's intent does not exist, CEGS cannot produce a satisfactory result.
- **Formal Synthesizer Scope**: The range of intents CEGS can correctly synthesize is dependent on the capabilities of the underlying Formal Synthesizer (NetComplete).
- **Intent Formalization**: The final configuration is correct only when the LLM correctly translates the natural language intent into a formal specification. While our evaluation showed high accuracy, this is dataset-specific.

## Contact

- **Primary Author**: Jianmin Liu
- **Corresponding Authors**: Dan Li, Yukai Miao
- **Institutions**: Tsinghua University, Zhongguancun Laboratory

For questions or suggestions, please contact us through GitHub Issues.

---

*This README provides an overview of the CEGS system. For more detailed information on the methodology and evaluation, please refer to the full paper.* 