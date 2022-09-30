![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20Windows-informational)
[![Python](https://img.shields.io/badge/Python-3.7%20%7C%203.8%20%7C%203.9%20%7C%203.10-informational)](https://www.python.org/)
[![Qiskit](https://img.shields.io/badge/Qiskit-Latest-6133BD)](https://github.com/Qiskit/qiskit)
[![License](https://img.shields.io/github/license/qiskit-community/dsm-swap?label=License)](LICENSE.txt)
[![Code style: Black](https://img.shields.io/badge/Code%20style-Black-000.svg)](https://github.com/psf/black)

<!-- ABOUT THIS PROJECT -->

# A doubly stochastic matrices-based approach to optimal qubit routing.


<!-- TABLE OF CONTENTS -->
### Table of Contents
* [Tutorials](docs)
* [About This Project](#about-this-project)
* [How to Give Feedback](#how-to-give-feedback)
* [Contribution Guidelines](#contribution-guidelines)
* [Acknowledgements](#acknowledgements)
* [References](#references)
* [License](#license)

---

### About This Project

Swap mapping is a quantum compiler optimization that by introducing SWAP gates maps a connectivity 
unconstrained circuit to an equivalent physically implementable one, which fulfills the 
hardware restrictions. Therefore, the placement of the SWAP gates can be interpreted as a discrete 
decision procedure. In this package, by introducing a structure called **doubly stochastic matrix** (DSM), 
which is defined as a convex combination of permutation matrices, we make the decision process 
smooth. Doubly stochastic matrices are contained in the Birkhoff polytope, in which the vertices 
represent single permutation matrices. In essence, the algorithm uses smooth constrained 
optimization to slide along the edges of the polytope toward the potential solutions on the 
vertices. Also, the algebraic structure of the cost function allows the minimization of both CNOT
count and circuit depth.

The software package includes a novelty visualization tool, based on braid diagrams, that highlights the effect of the swap mapping on the logical qubits. The figure below depicts an example for a Quantum Volume circuit with 8 qubits. Here the black wires represent the logical qubits and the red arcs stand for generic 2-qubit gates.
![Braids diagram](/docs/images/braids.png)

---
### How to use
The package can be installed from sources, please refer to [the contributing guideline](CONTRIBUTING.md).
Once it is installed it is available as a routing plugin in Qiskit. To transpile a circuit using this
plugin you have to pass ``routing_method="dsm"`` to the ``transpile`` function. Here is an example 
how to transpile a circuit to a ring-based coupling map. The algorithm works better when ``sabre``
is used for layout.

```python
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap

qc = QuantumCircuit.from_qasm_file("<a path to a circuit>")
cmap = CouplingMap.from_ring(qc.num_qubits, bidirectional=True)

tqc = transpile(qc, coupling_map=cmap, layout_method="sabre", routing_method="dsm", seed_transpiler=123, optimization_level=1)
print(tqc)
```

---

<!-- HOW TO GIVE FEEDBACK -->
### How to Give Feedback
We encourage your feedback! You can share your thoughts with us by:
- [Opening an issue](https://github.com/qiskit-community/dsm-swap/issues) in the repository
- [Starting a conversation on GitHub Discussions](https://github.com/qiskit-community/dsm-swap/discussions)

---

<!-- CONTRIBUTION GUIDELINES -->
### Contribution Guidelines
For information on how to contribute to this project, please take a look at [CONTRIBUTING.MD](CONTRIBUTING.md).

---

<!-- ACKNOWLEDGEMENTS -->
### Acknowledgements
This module is based on the theory and experiment described in [[1]](#references).

The code on which this module is based was written by Nicola Mariella, Anton Dekusar, and Albert Akhriev.

---

<!-- REFERENCES -->
### References
[1] Nicola Mariella, Sergiy Zhuk, *A doubly stochastic matrices-based approach to optimal qubit routing*, **(to be released soon)**.

<!-- LICENSE -->
### License
[Apache License 2.0](LICENSE.txt)