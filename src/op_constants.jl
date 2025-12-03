"""
    PAULI_X

Sparse Pauli X (σₓ) matrix: [[0, 1], [1, 0]]

The bit-flip operator. Exchanges |0⟩ ↔ |1⟩.
"""
const PAULI_X = [0 1; 1 0]
const SPIN_X = PAULI_X/2

"""
    PAULI_Y

Sparse Pauli Y (σᵧ) matrix: [[0, -i], [i, 0]]

The phase-flip and bit-flip operator.
"""
const PAULI_Y = [0 -im; im 0]
const SPIN_Y = PAULI_Y/2

"""
    PAULI_Z

Sparse Pauli Z (σᵧ) matrix: [[1, 0], [0, -1]]

The phase-flip operator. Maps |0⟩ → |0⟩, |1⟩ → -|1⟩.
"""
const PAULI_Z = [1 0; 0 -1]
const SPIN_Z = PAULI_Z/2

"""
    OCC_PART

Sparse occupation number operator for particles: [[1, 0], [0, 0]]

Projects onto the occupied state |1⟩.
"""
const OCC_PART = [1 0; 0 0]

"""
    OCC_HOLE

Sparse occupation number operator for holes: [[0, 0], [0, 1]]

Projects onto the empty state |0⟩.
"""
const OCC_HOLE = [0 0; 0 1]

"""
    RAISE

Sparse raising (creation) operator: [[0, 1], [0, 0]]

Maps |0⟩ → |1⟩, |1⟩ → 0. Also called σ₊ or a†.
"""
const RAISE = [0 1; 0 0]

"""
    LOWER

Sparse lowering (annihilation) operator: [[0, 0], [1, 0]]

Maps |1⟩ → |0⟩, |0⟩ → 0. Also called σ₋ or a.
"""
const LOWER = [0 0; 1 0]