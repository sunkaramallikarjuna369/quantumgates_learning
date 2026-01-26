"""
Unitary & Reversibility - Interactive Python Demonstrations
==========================================================

This script demonstrates unitary properties:
- Unitarity condition
- Probability conservation
- Reversibility
- Eigenvalue properties

Requirements: numpy, matplotlib
Install: pip install numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Common gates
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)


def is_unitary(matrix, tol=1e-10):
    """Check if a matrix is unitary."""
    n = matrix.shape[0]
    product = np.dot(matrix.conj().T, matrix)
    return np.allclose(product, np.eye(n), atol=tol)


def demo_unitarity_condition():
    """Demonstrate the unitarity condition."""
    print("=" * 60)
    print("UNITARITY CONDITION: U†U = UU† = I")
    print("=" * 60)
    
    gates = [
        ('Identity (I)', I),
        ('Pauli-X', X),
        ('Pauli-Y', Y),
        ('Pauli-Z', Z),
        ('Hadamard (H)', H),
        ('Phase (S)', S),
        ('T gate', T),
    ]
    
    print("\n--- Checking Unitarity of Common Gates ---")
    for name, gate in gates:
        product = np.dot(gate.conj().T, gate)
        is_unit = np.allclose(product, I)
        print(f"\n{name}:")
        print(f"  U†U = \n{np.round(product, 4)}")
        print(f"  Is unitary: {is_unit}")


def demo_probability_conservation():
    """Demonstrate probability conservation."""
    print("\n" + "=" * 60)
    print("PROBABILITY CONSERVATION")
    print("=" * 60)
    
    print("\nUnitary operations preserve the norm of state vectors.")
    print("||U|ψ⟩||² = ||ψ⟩||² = 1")
    
    # Create a random normalized state
    np.random.seed(42)
    state = np.random.randn(2) + 1j * np.random.randn(2)
    state = state / np.linalg.norm(state)
    state = state.reshape(2, 1)
    
    print(f"\n--- Initial State ---")
    print(f"|ψ⟩ = {state.flatten()}")
    print(f"||ψ⟩||² = {np.abs(np.vdot(state, state)):.6f}")
    
    print(f"\n--- After Various Gates ---")
    for name, gate in [('X', X), ('H', H), ('S', S), ('T', T)]:
        new_state = np.dot(gate, state)
        norm_sq = np.abs(np.vdot(new_state, new_state))
        print(f"After {name}: ||U|ψ⟩||² = {norm_sq:.6f}")


def demo_reversibility():
    """Demonstrate reversibility of unitary operations."""
    print("\n" + "=" * 60)
    print("REVERSIBILITY: U⁻¹ = U†")
    print("=" * 60)
    
    print("\nFor unitary matrices, the inverse equals the conjugate transpose.")
    
    # Create a random state
    np.random.seed(42)
    state = np.random.randn(2) + 1j * np.random.randn(2)
    state = state / np.linalg.norm(state)
    state = state.reshape(2, 1)
    
    print(f"\n--- Original State ---")
    print(f"|ψ⟩ = {np.round(state.flatten(), 4)}")
    
    # Apply H, then H† to reverse
    after_H = np.dot(H, state)
    print(f"\n--- After H ---")
    print(f"H|ψ⟩ = {np.round(after_H.flatten(), 4)}")
    
    recovered = np.dot(H.conj().T, after_H)
    print(f"\n--- After H† (reverse) ---")
    print(f"H†H|ψ⟩ = {np.round(recovered.flatten(), 4)}")
    print(f"Recovered original? {np.allclose(recovered, state)}")
    
    # More complex example
    print("\n--- Complex Circuit Reversal ---")
    circuit = np.dot(T, np.dot(H, np.dot(S, X)))
    after_circuit = np.dot(circuit, state)
    
    # Reverse: X† S† H† T†
    reverse_circuit = np.dot(X.conj().T, np.dot(S.conj().T, np.dot(H.conj().T, T.conj().T)))
    recovered = np.dot(reverse_circuit, after_circuit)
    
    print(f"After X·S·H·T: {np.round(after_circuit.flatten(), 4)}")
    print(f"After reversal: {np.round(recovered.flatten(), 4)}")
    print(f"Recovered original? {np.allclose(recovered, state)}")


def demo_inner_product_preservation():
    """Demonstrate inner product preservation."""
    print("\n" + "=" * 60)
    print("INNER PRODUCT PRESERVATION")
    print("=" * 60)
    
    print("\nUnitary operations preserve inner products: ⟨Uψ|Uφ⟩ = ⟨ψ|φ⟩")
    
    # Create two random states
    np.random.seed(42)
    psi = np.random.randn(2) + 1j * np.random.randn(2)
    psi = psi / np.linalg.norm(psi)
    psi = psi.reshape(2, 1)
    
    phi = np.random.randn(2) + 1j * np.random.randn(2)
    phi = phi / np.linalg.norm(phi)
    phi = phi.reshape(2, 1)
    
    original_inner = np.vdot(psi, phi)
    print(f"\n⟨ψ|φ⟩ = {original_inner:.4f}")
    
    print("\n--- After Various Gates ---")
    for name, gate in [('X', X), ('H', H), ('S', S), ('T', T)]:
        new_psi = np.dot(gate, psi)
        new_phi = np.dot(gate, phi)
        new_inner = np.vdot(new_psi, new_phi)
        preserved = np.allclose(new_inner, original_inner)
        print(f"After {name}: ⟨Uψ|Uφ⟩ = {new_inner:.4f} (preserved: {preserved})")


def demo_eigenvalues():
    """Demonstrate eigenvalue properties of unitary matrices."""
    print("\n" + "=" * 60)
    print("EIGENVALUE PROPERTIES")
    print("=" * 60)
    
    print("\nAll eigenvalues of unitary matrices have magnitude 1.")
    print("They lie on the unit circle in the complex plane.")
    
    gates = [
        ('Pauli-X', X),
        ('Pauli-Y', Y),
        ('Pauli-Z', Z),
        ('Hadamard', H),
        ('Phase (S)', S),
        ('T gate', T),
    ]
    
    for name, gate in gates:
        eigenvalues, _ = np.linalg.eig(gate)
        magnitudes = np.abs(eigenvalues)
        print(f"\n{name}:")
        print(f"  Eigenvalues: {np.round(eigenvalues, 4)}")
        print(f"  Magnitudes: {np.round(magnitudes, 4)}")
        print(f"  All on unit circle: {np.allclose(magnitudes, 1)}")


def demo_non_unitary():
    """Show examples of non-unitary matrices."""
    print("\n" + "=" * 60)
    print("NON-UNITARY MATRICES (NOT VALID GATES)")
    print("=" * 60)
    
    non_unitary = [
        ('[[1,1],[0,1]]', np.array([[1, 1], [0, 1]], dtype=complex)),
        ('[[2,0],[0,1]]', np.array([[2, 0], [0, 1]], dtype=complex)),
        ('[[1,0],[1,0]]', np.array([[1, 0], [1, 0]], dtype=complex)),
    ]
    
    for name, matrix in non_unitary:
        product = np.dot(matrix.conj().T, matrix)
        is_unit = is_unitary(matrix)
        print(f"\n{name}:")
        print(f"  M†M = \n{np.round(product, 4)}")
        print(f"  Is unitary: {is_unit}")
        print(f"  Cannot be a quantum gate!")


def plot_eigenvalues_unit_circle():
    """Plot eigenvalues on the unit circle."""
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#0a0a0a')
    
    # Draw unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'w-', alpha=0.3, linewidth=2)
    
    # Plot eigenvalues of various gates
    gates = [
        ('X', X, '#ff6b6b'),
        ('Y', Y, '#4ecdc4'),
        ('Z', Z, '#ffd93d'),
        ('H', H, '#64ffda'),
        ('S', S, '#ff8e8e'),
        ('T', T, '#a78bfa'),
    ]
    
    for name, gate, color in gates:
        eigenvalues, _ = np.linalg.eig(gate)
        for ev in eigenvalues:
            ax.scatter(ev.real, ev.imag, s=200, c=color, label=name, zorder=5)
            ax.annotate(name, (ev.real, ev.imag), textcoords="offset points", 
                       xytext=(10, 10), color=color, fontsize=10)
    
    ax.axhline(y=0, color='white', alpha=0.3)
    ax.axvline(x=0, color='white', alpha=0.3)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.set_xlabel('Real', color='white')
    ax.set_ylabel('Imaginary', color='white')
    ax.set_title('Eigenvalues of Unitary Gates on Unit Circle', 
                fontsize=14, fontweight='bold', color='#64ffda')
    ax.tick_params(colors='white')
    
    plt.tight_layout()
    plt.savefig('eigenvalues_unit_circle.png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    print("\nSaved: eigenvalues_unit_circle.png")
    plt.show()


def main():
    """Main function."""
    print("\n" + "=" * 60)
    print("UNITARY & REVERSIBILITY")
    print("Fundamental Properties of Quantum Operations")
    print("=" * 60)
    
    demo_unitarity_condition()
    demo_probability_conservation()
    demo_reversibility()
    demo_inner_product_preservation()
    demo_eigenvalues()
    demo_non_unitary()
    
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    plot_eigenvalues_unit_circle()
    
    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("""
1. All quantum gates must be UNITARY: U†U = I
2. Unitarity ensures PROBABILITY CONSERVATION
3. Unitary operations are REVERSIBLE: U⁻¹ = U†
4. Inner products are PRESERVED: ⟨Uψ|Uφ⟩ = ⟨ψ|φ⟩
5. All eigenvalues have MAGNITUDE 1 (unit circle)
6. Determinant has magnitude 1: |det(U)| = 1
7. Measurement is the ONLY irreversible operation
8. Non-unitary matrices CANNOT be quantum gates
    """)


if __name__ == "__main__":
    main()
