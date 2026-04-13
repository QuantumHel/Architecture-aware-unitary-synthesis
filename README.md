# Architecture-aware-unitary-synthesis
Python code for the architecture-aware unitary synthesis

# Usage
Make sure you have Python 3.11 and pip installed.

# Install the required packages

`pip install -r requirements.txt`

# Running the synthesis

`python ./unitary_synthesis.py --qmin [Minimum amount of qubits] --qmax [Maximum amount of qubits] --arch [Specify the architecture] --equiv [Check synthesized unitary correctness]`

Currently supported architectures are "garnet" and "marrakesh"
