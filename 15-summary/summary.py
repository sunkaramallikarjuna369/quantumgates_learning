"""
Summary - Complete Reference for Quantum Gates
==============================================

This script provides a complete reference for all quantum gates
covered in this learning platform.

Requirements: numpy, matplotlib
Install: pip install numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# SINGLE-QUBIT GATES
# ============================================================

# Identity
I = np.eye(2, dtype=complex)

# Pauli Gates
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

# Hadamard
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

# Phase Gates
S = np.array([[1, 0], [0, 1j]], dtype=complex)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)


def Rx(theta):
    """Rotation around X-axis."""
    return np.array([
        [np.cos(theta/2), -1j*np.sin(theta/2)],
        [-1j*np.sin(theta/2), np.cos(theta/2)]
    ], dtype=complex)


def Ry(theta):
    """Rotation around Y-axis."""
    return np.array([
        [np.cos(theta/2), -np.sin(theta/2)],
        [np.sin(theta/2), np.cos(theta/2)]
    ], dtype=complex)


def Rz(theta):
    """Rotation around Z-axis."""
    return np.array([
        [np.exp(-1j*theta/2), 0],
        [0, np.exp(1j*theta/2)]
    ], dtype=complex)


# ============================================================
# MULTI-QUBIT GATES
# ============================================================

CNOT = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
], dtype=complex)

CZ = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, -1]
], dtype=complex)

SWAP = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
], dtype=complex)

TOFFOLI = np.eye(8, dtype=complex)
TOFFOLI[6, 6] = 0
TOFFOLI[6, 7] = 1
TOFFOLI[7, 6] = 1
TOFFOLI[7, 7] = 0


def tensor(*args):
    """Compute tensor product of multiple matrices."""
    result = args[0]
    for m in args[1:]:
        result = np.kron(result, m)
    return result


def is_unitary(matrix, tol=1e-10):
    """Check if a matrix is unitary."""
    n = matrix.shape[0]
    product = np.dot(matrix.conj().T, matrix)
    return np.allclose(product, np.eye(n), atol=tol)


def print_gate_reference():
    """Print complete gate reference."""
    print("=" * 70)
    print("QUANTUM GATES - COMPLETE REFERENCE")
    print("=" * 70)
    
    print("\n" + "=" * 70)
    print("SINGLE-QUBIT GATES")
    print("=" * 70)
    
    single_gates = [
        ('Identity (I)', I, 'Does nothing'),
        ('Pauli-X', X, 'Bit flip: |0⟩↔|1⟩'),
        ('Pauli-Y', Y, 'Bit + phase flip'),
        ('Pauli-Z', Z, 'Phase flip: |1⟩→-|1⟩'),
        ('Hadamard (H)', H, 'Creates superposition'),
        ('Phase (S)', S, 'π/2 phase rotation'),
        ('T gate', T, 'π/4 phase rotation'),
    ]
    
    for name, gate, description in single_gates:
        print(f"\n--- {name} ---")
        print(f"Description: {description}")
        print(f"Matrix:\n{np.round(gate, 4)}")
        print(f"Unitary: {is_unitary(gate)}")
    
    print("\n" + "=" * 70)
    print("ROTATION GATES")
    print("=" * 70)
    
    print("\nRx(θ) - Rotation around X-axis")
    print(f"Rx(π/2):\n{np.round(Rx(np.pi/2), 4)}")
    
    print("\nRy(θ) - Rotation around Y-axis")
    print(f"Ry(π/2):\n{np.round(Ry(np.pi/2), 4)}")
    
    print("\nRz(θ) - Rotation around Z-axis")
    print(f"Rz(π/2):\n{np.round(Rz(np.pi/2), 4)}")
    
    print("\n" + "=" * 70)
    print("MULTI-QUBIT GATES")
    print("=" * 70)
    
    multi_gates = [
        ('CNOT', CNOT, 'Flips target if control=|1⟩'),
        ('CZ', CZ, 'Phase flip if both=|1⟩'),
        ('SWAP', SWAP, 'Exchanges two qubits'),
    ]
    
    for name, gate, description in multi_gates:
        print(f"\n--- {name} ---")
        print(f"Description: {description}")
        print(f"Matrix:\n{gate.astype(int)}")
        print(f"Unitary: {is_unitary(gate)}")
    
    print("\n--- Toffoli (CCNOT) ---")
    print("Description: Flips target if both controls=|1⟩")
    print(f"Matrix (8×8):\n{TOFFOLI.astype(int)}")
    print(f"Unitary: {is_unitary(TOFFOLI)}")


def print_key_formulas():
    """Print key formulas and relationships."""
    print("\n" + "=" * 70)
    print("KEY FORMULAS AND RELATIONSHIPS")
    print("=" * 70)
    
    print("\n--- Euler Decomposition ---")
    print("Any single-qubit gate U = Rz(α) · Ry(β) · Rz(γ)")
    
    print("\n--- Gate Relationships ---")
    print("X² = Y² = Z² = I")
    print("H² = I")
    print("S² = Z")
    print("T² = S")
    print("XYZ = iI")
    
    print("\n--- Decompositions ---")
    print("SWAP = CNOT₁₂ · CNOT₂₁ · CNOT₁₂")
    print("CZ = (I⊗H) · CNOT · (I⊗H)")
    print("X = H · Z · H")
    
    print("\n--- Bell States ---")
    print("|Φ⁺⟩ = (|00⟩ + |11⟩)/√2")
    print("|Φ⁻⟩ = (|00⟩ - |11⟩)/√2")
    print("|Ψ⁺⟩ = (|01⟩ + |10⟩)/√2")
    print("|Ψ⁻⟩ = (|01⟩ - |10⟩)/√2")
    
    print("\n--- Universal Gate Sets ---")
    print("1. Clifford + T: {H, S, CNOT, T}")
    print("2. CNOT + Rotations: {CNOT, Rx, Ry, Rz}")
    print("3. Toffoli + H: {Toffoli, H}")


def verify_relationships():
    """Verify key gate relationships."""
    print("\n" + "=" * 70)
    print("VERIFYING RELATIONSHIPS")
    print("=" * 70)
    
    print("\n--- X² = I ---")
    print(f"X² = \n{np.round(np.dot(X, X), 4)}")
    print(f"Equals I: {np.allclose(np.dot(X, X), I)}")
    
    print("\n--- S² = Z ---")
    print(f"S² = \n{np.round(np.dot(S, S), 4)}")
    print(f"Equals Z: {np.allclose(np.dot(S, S), Z)}")
    
    print("\n--- T² = S ---")
    print(f"T² = \n{np.round(np.dot(T, T), 4)}")
    print(f"Equals S: {np.allclose(np.dot(T, T), S)}")
    
    print("\n--- X = H · Z · H ---")
    result = np.dot(H, np.dot(Z, H))
    print(f"H·Z·H = \n{np.round(result, 4)}")
    print(f"Equals X: {np.allclose(result, X)}")
    
    print("\n--- CZ = (I⊗H) · CNOT · (I⊗H) ---")
    IH = tensor(I, H)
    result = np.dot(IH, np.dot(CNOT, IH))
    print(f"Equals CZ: {np.allclose(result, CZ)}")


def plot_gate_summary():
    """Create a visual summary of all gates."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.patch.set_facecolor('#1a1a2e')
    fig.suptitle('Quantum Gates Visual Summary', fontsize=16, fontweight='bold', color='white')
    
    gates = [
        ('Pauli-X', X, '#ff6b6b'),
        ('Pauli-Y', Y, '#4ecdc4'),
        ('Pauli-Z', Z, '#ffd93d'),
        ('Hadamard', H, '#64ffda'),
        ('Phase (S)', S, '#a78bfa'),
        ('T Gate', T, '#ff8e8e'),
        ('CNOT', CNOT, '#667eea'),
        ('SWAP', SWAP, '#764ba2'),
    ]
    
    for ax, (name, gate, color) in zip(axes.flatten(), gates):
        ax.set_facecolor('#0a0a0a')
        
        # Plot matrix magnitude
        magnitude = np.abs(gate)
        im = ax.imshow(magnitude, cmap='viridis', vmin=0, vmax=1)
        
        # Add text annotations
        for i in range(gate.shape[0]):
            for j in range(gate.shape[1]):
                val = gate[i, j]
                if np.abs(val) > 0.01:
                    text = f'{val.real:.1f}' if val.imag == 0 else f'{val:.1f}'
                    ax.text(j, i, text, ha='center', va='center', 
                           color='white', fontsize=8 if gate.shape[0] > 2 else 10)
        
        ax.set_title(name, color=color, fontsize=12, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig('gates_summary.png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    print("\nSaved: gates_summary.png")
    plt.show()


def main():
    """Main function."""
    print_gate_reference()
    print_key_formulas()
    verify_relationships()
    
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    
    plot_gate_summary()
    
    print("\n" + "=" * 70)
    print("CONGRATULATIONS!")
    print("=" * 70)
    print("""
You have completed the Quantum Gates Learning Platform!

Key concepts mastered:
1. Single-qubit gates (X, Y, Z, H, S, T)
2. Rotation gates (Rx, Ry, Rz)
3. Multi-qubit gates (CNOT, CZ, SWAP, Toffoli)
4. Unitarity and reversibility
5. Universal gate sets
6. Gate decomposition
7. Bloch sphere visualization
8. Entanglement and Bell states

Next steps:
- Practice with the exercises
- Implement quantum algorithms
- Explore quantum error correction
- Learn about quantum hardware
    """)


if __name__ == "__main__":
    main()
