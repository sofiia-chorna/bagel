# Concept-Based Mechanistic Interpretability Using Structured Knowledge Graphs

This repository contains scripts to reproduce results from the paper "Concept-Based Mechanistic Interpretability Using Structured Knowledge Graphs" (preprint: https://arxiv.org/abs/2507.05810).

Check our website with the results: https://knowledge-graph-ui-4a7cb5.gitlab.io/

## Abstract
While concept-based interpretability methods have traditionally focused on local explanations of neural network predictions, we propose a novel framework and interactive tool that extends these methods into the domain of mechanistic interpretability. Our approach enables a global dissection of model behavior by analyzing how high-level semantic attributes (referred to as concepts) emerge, interact, and propagate through internal model components. Unlike prior work that isolates individual neurons or predictions, our framework systematically quantifies how semantic concepts are represented across layers, revealing latent circuits and information flow that underlie model decision-making. A key innovation is our visualization platform that we named BAGEL (for Bias Analysis with a Graph for global Explanation Layers), which presents these insights in a structured knowledge graph, allowing users to explore concept-class relationships, identify spurious correlations, and enhance model trustworthiness. Our framework is model-agnostic, scalable, and contributes to a deeper understanding of how deep learning models generalize (or fail to) in the presence of dataset biases.

<p align="center">
<img height="300" alt="method drawio" src="https://github.com/user-attachments/assets/55438dbc-90f1-4721-8ca0-5504a720c329" />
<p>

## Citing
```
@misc{chorna2025conceptbasedinterpretability,
      title={Concept-Based Mechanistic Interpretability Using Structured Knowledge Graphs}, 
      author={Sofiia Chorna and Kateryna Tarelkina and Eloïse Berthier and Gianni Franchi},
      year={2025},
      eprint={2507.05810},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2507.05810}, 
}
```
