# Order Effects in Binary Judgments

This repository contains the data and Python code required to reproduce the empirical analyses and model-based results reported in the paper:

**Order Effects in Binary Judgments: A Minimal Hamiltonian Quantum-Like Model**

The study investigates order effects in binary evaluative judgments and demonstrates how non-commutativity arises under attribute conflict using a minimal Hamiltonian quantum-like framework.

---

## Repository Contents

- `data/`  
  Empirical data collected via an online experiment (Excel format).

- `code/`  
  Python scripts implementing:
  - computation of empirical sequential probabilities  
  - quantum-like sequential measurement using Lüders’ rule  
  - parameter estimation for the minimal Hamiltonian model  

---

## Requirements

The analysis was implemented in Python (≥3.9). Required packages:

- numpy  
- pandas  
- scipy  
- openpyxl  

To install dependencies:

```bash
pip install -r main/requirements.txt

