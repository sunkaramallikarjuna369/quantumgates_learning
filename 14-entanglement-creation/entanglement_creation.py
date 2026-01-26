"""
Entanglement Creation - Interactive Python Demonstrations
========================================================

This script demonstrates entanglement creation:
- Bell state creation
- Entanglement verification
- Measurement correlations
- Applications of entanglement

Requirements: numpy, matplotlib
Install: pip install numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Basis states
ket_0 = np.array([[1], [0]], dtype=complex)
ket_1 = np.array([[0], [1]], dtype=complex)

# Gates
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

CNOT = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
], dtype=complex)


def tensor(*args):
    """Compute tensor product of multiple matrices."""
    result = args[0]
    for m in args[1:]:
        result = np.kron(result, m)
    return result


def is_separable(state, tol=1e-10):
    """Check if a two-qubit state is separable (not entangled)."""
    # Reshape to 2x2 matrix
    matrix = state.reshape(2, 2)
    
    # Check if rank is 1 (separable states have rank-1 matrix)
    _, s, _ = np.linalg.svd(matrix)
    return np.sum(s > tol) == 1


def create_bell_state(which='phi+'):
    """Create one of the four Bell states."""
    # Start with |00⟩
    state = tensor(ket_0, ket_0)
    
    # Apply H to first qubit
    H_I = tensor(H, I)
    state = np.dot(H_I, state)
    
    # Apply CNOT
    state = np.dot(CNOT, state)
    
    # This gives |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
    
    if which == 'phi+':
        return state
    elif which == 'phi-':
        # Apply Z to first qubit: (|00⟩ - |11⟩)/√2
        Z_I = tensor(Z, I)
        return np.dot(Z_I, state)
    elif which == 'psi+':
        # Apply X to second qubit: (|01⟩ + |10⟩)/√2
        I_X = tensor(I, X)
        return np.dot(I_X, state)
    elif which == 'psi-':
        # Apply X to second, Z to first: (|01⟩ - |10⟩)/√2
        I_X = tensor(I, X)
        Z_I = tensor(Z, I)
        return np.dot(Z_I, np.dot(I_X, state))
    else:
        raise ValueError(f"Unknown Bell state: {which}")


def demo_bell_state_creation():
    """Demonstrate Bell state creation."""
    print("=" * 60)
    print("BELL STATE CREATION")
    print("=" * 60)
    
    print("\nCircuit: H on qubit 1, then CNOT")
    print("|00⟩ → H⊗I → (|0⟩+|1⟩)|0⟩/√2 → CNOT → (|00⟩+|11⟩)/√2")
    
    # Step by step
    print("\n--- Step by Step ---")
    
    state = tensor(ket_0, ket_0)
    print(f"Initial |00⟩: {state.flatten()}")
    
    H_I = tensor(H, I)
    state = np.dot(H_I, state)
    print(f"After H⊗I: {np.round(state.flatten(), 4)}")
    
    state = np.dot(CNOT, state)
    print(f"After CNOT: {np.round(state.flatten(), 4)}")
    
    print("\nThis is |Φ⁺⟩ = (|00⟩ + |11⟩)/√2")


def demo_all_bell_states():
    """Demonstrate all four Bell states."""
    print("\n" + "=" * 60)
    print("THE FOUR BELL STATES")
    print("=" * 60)
    
    bell_states = [
        ('|Φ⁺⟩', 'phi+', '(|00⟩ + |11⟩)/√2'),
        ('|Φ⁻⟩', 'phi-', '(|00⟩ - |11⟩)/√2'),
        ('|Ψ⁺⟩', 'psi+', '(|01⟩ + |10⟩)/√2'),
        ('|Ψ⁻⟩', 'psi-', '(|01⟩ - |10⟩)/√2'),
    ]
    
    for name, key, formula in bell_states:
        state = create_bell_state(key)
        print(f"\n{name} = {formula}")
        print(f"  Amplitudes: {np.round(state.flatten(), 4)}")
        print(f"  Separable: {is_separable(state)}")


def demo_measurement_correlations():
    """Demonstrate measurement correlations in Bell states."""
    print("\n" + "=" * 60)
    print("MEASUREMENT CORRELATIONS")
    print("=" * 60)
    
    print("\nFor |Φ⁺⟩ = (|00⟩ + |11⟩)/√2:")
    state = create_bell_state('phi+')
    
    probs = np.abs(state.flatten())**2
    print(f"\nProbabilities:")
    print(f"  P(|00⟩) = {probs[0]:.2%}")
    print(f"  P(|01⟩) = {probs[1]:.2%}")
    print(f"  P(|10⟩) = {probs[2]:.2%}")
    print(f"  P(|11⟩) = {probs[3]:.2%}")
    
    print("\nCorrelation: Both qubits ALWAYS measure the same!")
    print("If qubit 1 = |0⟩, then qubit 2 = |0⟩ (100%)")
    print("If qubit 1 = |1⟩, then qubit 2 = |1⟩ (100%)")
    
    print("\n--- For |Ψ⁺⟩ = (|01⟩ + |10⟩)/√2: ---")
    state = create_bell_state('psi+')
    probs = np.abs(state.flatten())**2
    print(f"\nProbabilities:")
    print(f"  P(|00⟩) = {probs[0]:.2%}")
    print(f"  P(|01⟩) = {probs[1]:.2%}")
    print(f"  P(|10⟩) = {probs[2]:.2%}")
    print(f"  P(|11⟩) = {probs[3]:.2%}")
    
    print("\nAnti-correlation: Qubits ALWAYS measure opposite!")


def demo_entanglement_verification():
    """Verify entanglement using separability test."""
    print("\n" + "=" * 60)
    print("ENTANGLEMENT VERIFICATION")
    print("=" * 60)
    
    print("\nA state is entangled if it CANNOT be written as |ψ₁⟩⊗|ψ₂⟩")
    
    # Product state
    product = tensor(
        (ket_0 + ket_1) / np.sqrt(2),  # |+⟩
        ket_0
    )
    
    # Bell state
    bell = create_bell_state('phi+')
    
    print("\n--- Product State |+⟩⊗|0⟩ ---")
    print(f"Amplitudes: {np.round(product.flatten(), 4)}")
    print(f"Separable: {is_separable(product)}")
    print("This is NOT entangled!")
    
    print("\n--- Bell State |Φ⁺⟩ ---")
    print(f"Amplitudes: {np.round(bell.flatten(), 4)}")
    print(f"Separable: {is_separable(bell)}")
    print("This IS entangled!")


def demo_bell_orthonormality():
    """Demonstrate that Bell states are orthonormal."""
    print("\n" + "=" * 60)
    print("BELL STATES ARE ORTHONORMAL")
    print("=" * 60)
    
    states = {
        '|Φ⁺⟩': create_bell_state('phi+'),
        '|Φ⁻⟩': create_bell_state('phi-'),
        '|Ψ⁺⟩': create_bell_state('psi+'),
        '|Ψ⁻⟩': create_bell_state('psi-'),
    }
    
    print("\nInner products ⟨ψᵢ|ψⱼ⟩:")
    print(f"{'':8s}", end='')
    for name in states:
        print(f"{name:8s}", end='')
    print()
    
    for name1, state1 in states.items():
        print(f"{name1:8s}", end='')
        for name2, state2 in states.items():
            inner = np.abs(np.vdot(state1, state2))**2
            print(f"{inner:8.2f}", end='')
        print()
    
    print("\nDiagonal = 1 (normalized), off-diagonal = 0 (orthogonal)")


def plot_bell_states():
    """Visualize Bell state amplitudes."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.patch.set_facecolor('#1a1a2e')
    fig.suptitle('The Four Bell States', fontsize=16, fontweight='bold', color='white')
    
    bell_states = [
        ('|Φ⁺⟩ = (|00⟩ + |11⟩)/√2', 'phi+'),
        ('|Φ⁻⟩ = (|00⟩ - |11⟩)/√2', 'phi-'),
        ('|Ψ⁺⟩ = (|01⟩ + |10⟩)/√2', 'psi+'),
        ('|Ψ⁻⟩ = (|01⟩ - |10⟩)/√2', 'psi-'),
    ]
    
    basis_labels = ['|00⟩', '|01⟩', '|10⟩', '|11⟩']
    colors = ['#64ffda', '#ff6b6b', '#ffd93d', '#a78bfa']
    
    for ax, (title, key), color in zip(axes.flatten(), bell_states, colors):
        ax.set_facecolor('#0a0a0a')
        
        state = create_bell_state(key)
        amplitudes = state.flatten().real
        
        bars = ax.bar(basis_labels, amplitudes, color=color, edgecolor='white', linewidth=2)
        ax.axhline(y=0, color='white', alpha=0.3)
        ax.set_ylim(-0.8, 0.8)
        ax.set_title(title, color='#64ffda', fontsize=12)
        ax.set_ylabel('Amplitude', color='white')
        ax.tick_params(colors='white')
        
        # Add value labels
        for bar, val in zip(bars, amplitudes):
            if abs(val) > 0.01:
                ax.text(bar.get_x() + bar.get_width()/2, val + 0.05 * np.sign(val),
                       f'{val:.2f}', ha='center', va='bottom' if val > 0 else 'top',
                       color='white', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('bell_states.png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    print("\nSaved: bell_states.png")
    plt.show()


def plot_entanglement_circuit():
    """Visualize the Bell state creation circuit."""
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#0a0a0a')
    
    # Draw qubit lines
    ax.plot([0, 10], [2, 2], color='#64ffda', linewidth=2)
    ax.plot([0, 10], [0, 0], color='#64ffda', linewidth=2)
    
    # Labels
    ax.text(-0.5, 2, '|0⟩', fontsize=14, color='#ffd93d', ha='right', va='center')
    ax.text(-0.5, 0, '|0⟩', fontsize=14, color='#ffd93d', ha='right', va='center')
    
    # Hadamard gate
    rect = plt.Rectangle((2, 1.5), 1, 1, fill=True, facecolor='#667eea', edgecolor='white', linewidth=2)
    ax.add_patch(rect)
    ax.text(2.5, 2, 'H', fontsize=14, color='white', ha='center', va='center', fontweight='bold')
    
    # CNOT gate
    ax.plot(5, 2, 'o', markersize=15, color='#ff6b6b')
    ax.plot([5, 5], [2, 0], color='#ff6b6b', linewidth=2)
    circle = plt.Circle((5, 0), 0.3, fill=False, color='#ff6b6b', linewidth=2)
    ax.add_patch(circle)
    ax.plot([4.7, 5.3], [0, 0], color='#ff6b6b', linewidth=2)
    
    # Output label
    ax.text(10.5, 1, '|Φ⁺⟩', fontsize=16, color='#4ecdc4', ha='left', va='center')
    
    # State annotations
    ax.text(1, 3, '|00⟩', fontsize=12, color='white', ha='center')
    ax.text(3.5, 3, '(|0⟩+|1⟩)|0⟩/√2', fontsize=10, color='white', ha='center')
    ax.text(7, 3, '(|00⟩+|11⟩)/√2', fontsize=10, color='white', ha='center')
    
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 4)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Bell State Creation Circuit', fontsize=14, fontweight='bold', color='#64ffda')
    
    plt.tight_layout()
    plt.savefig('bell_circuit.png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    print("Saved: bell_circuit.png")
    plt.show()


def main():
    """Main function."""
    print("\n" + "=" * 60)
    print("ENTANGLEMENT CREATION")
    print("Creating and Understanding Bell States")
    print("=" * 60)
    
    demo_bell_state_creation()
    demo_all_bell_states()
    demo_measurement_correlations()
    demo_entanglement_verification()
    demo_bell_orthonormality()
    
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    plot_bell_states()
    plot_entanglement_circuit()
    
    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("""
1. Bell states are maximally entangled two-qubit states
2. Created by H + CNOT circuit
3. Four Bell states: |Φ⁺⟩, |Φ⁻⟩, |Ψ⁺⟩, |Ψ⁻⟩
4. They form an orthonormal basis
5. Measuring one qubit determines the other
6. Cannot be written as product states
7. Enable quantum teleportation and cryptography
8. Foundation for quantum computing advantage
    """)


if __name__ == "__main__":
    main()
