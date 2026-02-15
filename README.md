# SPC-Trajectory-Prediction-Framework

This repository contains the official implementation of the paper **"Enhancing Safety in Long-Tail Scenarios: An Interpretable Trajectory Prediction Framework with LLM-Driven Semantic-Physical Consistency"**.

> **⚠️ Note:** We are currently organizing and cleaning the source code to ensure it is user-friendly and fully reproducible. The complete replication package, including detailed instructions, will be released shortly.

## 📖 Abstract
Trajectory prediction is pivotal for autonomous driving stability, yet existing deep learning models suffer from opaque decision logic. This paper proposes a **Large Language Model (LLM)-driven hierarchical framework** that resolves the conflict between interpretability and physical compliance by constructing a **semantic-physical consistency**.

Our method effectively bridges discrete semantic space with continuous physical space using a learnable embedding matrix and a CoT reasoning process.

## 🚀 Key Features
- **LLM-Driven CoT Reasoning**: Utilizes structured prompt templates to perform Chain-of-Thought reasoning for interpretable intentions.
- **Semantic-Physical Consistency**: A hierarchical framework that decouples prediction into upper-level logical reasoning and lower-level dynamics generation to ensure safety.
- **SOTA Performance**: Verified on the **DeepAccident dataset**, reducing the Accident Average Displacement Error (ADE) by **13.8%** and lowering the collision rate to **1.25%** in Out-of-Distribution scenarios.

## 📂 Framework Structure (Coming Soon)
The code will be organized into the following modules:
- `Multi-modal Feature Encoding`: Heterogeneous vectorization for agent states and map topology.
- `Hierarchical Interaction Modeling`: Dual-stacked Transformer for agent-agent and agent-map interactions.
- `LLM Intention Decision`: Interface for LLM-based semantic reasoning and meta-action generation.
- `Constrained Trajectory Generation`: Physics-based decoder with safety constraints.

## 🛠️ Prerequisites
- Python 3.8+
- PyTorch
- Transformers
- DeepAccident Dataset (See Data Preparation)

## 💾 Data Preparation
This framework is evaluated on the **DeepAccident** dataset.
Due to license and size constraints, please download the dataset from the official repository:
[**DeepAccident Download Page**](https://deepaccident.github.io/download.html)


