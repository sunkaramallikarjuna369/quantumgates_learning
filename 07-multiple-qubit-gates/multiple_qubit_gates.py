"""
Multiple-Qubit Gates - Interactive Python Demonstrations
=======================================================

This script demonstrates multi-qubit systems:
- Tensor products
- Two-qubit state spaces
- Product vs entangled states
- Multi-qubit gate operations

Requirements: numpy, matplotlib
Install: pip install numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

# Single-qubit basis states
ket_0 = np.array([[1], [0]], dtype=complex)
ket_1 = np.array([[0], [1]], dtype=complex)

# Single-qubit gates
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def tensor(a, b):
    """Compute tensor product of two matrices/vectors."""
    return np.kron(a, b)


def create_two_qubit_state(q1, q2):
    """Create two-qubit state from individual qubit states."""
    return tensor(q1, q2)


def demo_tensor_product():
    """Demonstrate tensor product basics."""
    print("=" * 60)
    print("TENSOR PRODUCT")
    print("=" * 60)
    
    print("\n--- Two-Qubit Basis States ---")
    
    ket_00 = tensor(ket_0, ket_0)
    ket_01 = tensor(ket_0, ket_1)
    ket_10 = tensor(ket_1, ket_0)
    ket_11 = tensor(ket_1, ket_1)
    
    print(f"\n|00⟩ = |0⟩ ⊗ |0⟩ = {ket_00.flatten()}")
    print(f"|01⟩ = |0⟩ ⊗ |1⟩ = {ket_01.flatten()}")
    print(f"|10⟩ = |1⟩ ⊗ |0⟩ = {ket_10.flatten()}")
    print(f"|11⟩ = |1⟩ ⊗ |1⟩ = {ket_11.flatten()}")
    
    print("\n--- Tensor Product of Superpositions ---")
    ket_plus = (ket_0 + ket_1) / np.sqrt(2)
    ket_plus_plus = tensor(ket_plus, ket_plus)
    print(f"\n|+⟩ ⊗ |+⟩ = |++⟩ = {np.round(ket_plus_plus.flatten(), 4)}")
    print("This is (|00⟩ + |01⟩ + |10⟩ + |11⟩)/2")
    
    return ket_00, ket_01, ket_10, ket_11


def demo_gate_tensor():
    """Demonstrate tensor product of gates."""
    print("\n" + "=" * 60)
    print("TENSOR PRODUCT OF GATES")
    print("=" * 60)
    
    print("\n--- X ⊗ I (X on first qubit, I on second) ---")
    X_I = tensor(X, I)
    print(f"\nX ⊗ I =\n{X_I.astype(int)}")
    
    print("\n--- I ⊗ X (I on first qubit, X on second) ---")
    I_X = tensor(I, X)
    print(f"\nI ⊗ X =\n{I_X.astype(int)}")
    
    print("\n--- H ⊗ H (Hadamard on both qubits) ---")
    H_H = tensor(H, H)
    print(f"\nH ⊗ H =\n{np.round(H_H, 4)}")
    
    print("\n--- Action of X ⊗ I on |00⟩ ---")
    ket_00 = tensor(ket_0, ket_0)
    result = np.dot(X_I, ket_00)
    print(f"(X ⊗ I)|00⟩ = {result.flatten()} = |10⟩")
    
    print("\n--- Action of I ⊗ X on |00⟩ ---")
    result = np.dot(I_X, ket_00)
    print(f"(I ⊗ X)|00⟩ = {result.flatten()} = |01⟩")


def demo_product_vs_entangled():
    """Demonstrate product states vs entangled states."""
    print("\n" + "=" * 60)
    print("PRODUCT STATES vs ENTANGLED STATES")
    print("=" * 60)
    
    print("\n--- Product States ---")
    print("Can be written as |ψ₁⟩ ⊗ |ψ₂⟩")
    
    # |00⟩
    ket_00 = tensor(ket_0, ket_0)
    print(f"\n|00⟩ = |0⟩ ⊗ |0⟩ = {ket_00.flatten()}")
    
    # |+0⟩
    ket_plus = (ket_0 + ket_1) / np.sqrt(2)
    ket_plus_0 = tensor(ket_plus, ket_0)
    print(f"|+0⟩ = |+⟩ ⊗ |0⟩ = {np.round(ket_plus_0.flatten(), 4)}")
    
    print("\n--- Entangled States (Bell States) ---")
    print("CANNOT be written as |ψ₁⟩ ⊗ |ψ₂⟩")
    
    ket_01 = tensor(ket_0, ket_1)
    ket_10 = tensor(ket_1, ket_0)
    ket_11 = tensor(ket_1, ket_1)
    
    # Bell states
    phi_plus = (ket_00 + ket_11) / np.sqrt(2)
    phi_minus = (ket_00 - ket_11) / np.sqrt(2)
    psi_plus = (ket_01 + ket_10) / np.sqrt(2)
    psi_minus = (ket_01 - ket_10) / np.sqrt(2)
    
    print(f"\n|Φ⁺⟩ = (|00⟩ + |11⟩)/√2 = {np.round(phi_plus.flatten(), 4)}")
    print(f"|Φ⁻⟩ = (|00⟩ - |11⟩)/√2 = {np.round(phi_minus.flatten(), 4)}")
    print(f"|Ψ⁺⟩ = (|01⟩ + |10⟩)/√2 = {np.round(psi_plus.flatten(), 4)}")
    print(f"|Ψ⁻⟩ = (|01⟩ - |10⟩)/√2 = {np.round(psi_minus.flatten(), 4)}")
    
    return phi_plus, phi_minus, psi_plus, psi_minus


def is_separable(state, tol=1e-10):
    """
    Check if a two-qubit state is separable (product state).
    Uses the Schmidt decomposition / partial trace method.
    """
    # Reshape to 2x2 matrix
    mat = state.reshape(2, 2)
    
    # Compute singular values
    _, s, _ = np.linalg.svd(mat)
    
    # Separable if only one non-zero singular value
    non_zero = np.sum(np.abs(s) > tol)
    return non_zero == 1


def demo_separability():
    """Demonstrate separability test."""
    print("\n" + "=" * 60)
    print("SEPARABILITY TEST")
    print("=" * 60)
    
    states = [
        ('|00⟩', tensor(ket_0, ket_0)),
        ('|+0⟩', tensor((ket_0 + ket_1)/np.sqrt(2), ket_0)),
        ('|++⟩', tensor((ket_0 + ket_1)/np.sqrt(2), (ket_0 + ket_1)/np.sqrt(2))),
        ('|Φ⁺⟩ = (|00⟩+|11⟩)/√2', (tensor(ket_0, ket_0) + tensor(ket_1, ket_1))/np.sqrt(2)),
        ('|Ψ⁺⟩ = (|01⟩+|10⟩)/√2', (tensor(ket_0, ket_1) + tensor(ket_1, ket_0))/np.sqrt(2)),
    ]
    
    print("\nTesting separability of various states:")
    for name, state in states:
        sep = is_separable(state)
        status = "PRODUCT (separable)" if sep else "ENTANGLED (not separable)"
        print(f"  {name}: {status}")


def demo_state_space_growth():
    """Demonstrate exponential growth of state space."""
    print("\n" + "=" * 60)
    print("EXPONENTIAL STATE SPACE GROWTH")
    print("=" * 60)
    
    print("\n--- State Space Dimensions ---")
    print(f"{'Qubits':<10} {'Dimensions':<15} {'Basis States':<30}")
    print("-" * 55)
    
    for n in range(1, 11):
        dim = 2**n
        if n <= 3:
            basis = ', '.join([format(i, f'0{n}b') for i in range(dim)])
            basis = f"|{basis.replace(', ', '⟩, |')}⟩"
        else:
            basis = f"|{'0'*n}⟩ to |{'1'*n}⟩"
        print(f"{n:<10} {dim:<15} {basis}")
    
    print("\n--- Memory Requirements ---")
    print("Each amplitude is a complex number (16 bytes)")
    for n in [10, 20, 30, 40, 50]:
        dim = 2**n
        bytes_needed = dim * 16
        if bytes_needed < 1024:
            mem = f"{bytes_needed} B"
        elif bytes_needed < 1024**2:
            mem = f"{bytes_needed/1024:.1f} KB"
        elif bytes_needed < 1024**3:
            mem = f"{bytes_needed/1024**2:.1f} MB"
        elif bytes_needed < 1024**4:
            mem = f"{bytes_needed/1024**3:.1f} GB"
        elif bytes_needed < 1024**5:
            mem = f"{bytes_needed/1024**4:.1f} TB"
        else:
            mem = f"{bytes_needed/1024**5:.1f} PB"
        print(f"  {n} qubits: {dim:,} amplitudes = {mem}")


def plot_two_qubit_states():
    """Visualize two-qubit state amplitudes."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.patch.set_facecolor('#1a1a2e')
    fig.suptitle('Two-Qubit State Amplitudes', fontsize=16, fontweight='bold', color='white')
    
    ket_00 = tensor(ket_0, ket_0)
    ket_01 = tensor(ket_0, ket_1)
    ket_10 = tensor(ket_1, ket_0)
    ket_11 = tensor(ket_1, ket_1)
    ket_plus = (ket_0 + ket_1) / np.sqrt(2)
    
    states = [
        ('|00⟩ (Product)', ket_00),
        ('|01⟩ (Product)', ket_01),
        ('|++⟩ (Product)', tensor(ket_plus, ket_plus)),
        ('|Φ⁺⟩ (Entangled)', (ket_00 + ket_11)/np.sqrt(2)),
        ('|Ψ⁺⟩ (Entangled)', (ket_01 + ket_10)/np.sqrt(2)),
        ('|Ψ⁻⟩ (Entangled)', (ket_01 - ket_10)/np.sqrt(2)),
    ]
    
    basis_labels = ['|00⟩', '|01⟩', '|10⟩', '|11⟩']
    
    for ax, (name, state) in zip(axes.flat, states):
        ax.set_facecolor('#0a0a0a')
        
        probs = np.abs(state.flatten())**2
        colors = ['#64ffda' if p > 0.01 else '#333' for p in probs]
        
        bars = ax.bar(basis_labels, probs, color=colors, edgecolor='white', linewidth=1)
        
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Probability', color='white')
        ax.set_title(name, fontsize=11, fontweight='bold', color='#64ffda')
        ax.tick_params(colors='white')
        
        for bar, prob in zip(bars, probs):
            if prob > 0.01:
                ax.text(bar.get_x() + bar.get_width()/2., prob + 0.02,
                       f'{prob:.2f}', ha='center', va='bottom', 
                       fontsize=10, color='white')
    
    plt.tight_layout()
    plt.savefig('two_qubit_states.png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    print("\nSaved: two_qubit_states.png")
    plt.show()


def plot_state_space_growth():
    """Visualize exponential state space growth."""
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#0a0a0a')
    
    n_qubits = np.arange(1, 31)
    dimensions = 2 ** n_qubits
    
    ax.semilogy(n_qubits, dimensions, 'o-', color='#64ffda', linewidth=2, markersize=6)
    ax.fill_between(n_qubits, 1, dimensions, alpha=0.2, color='#64ffda')
    
    # Highlight classical limits
    ax.axhline(y=2**30, color='#ff6b6b', linestyle='--', alpha=0.5, label='~1 billion (classical limit)')
    ax.axhline(y=2**50, color='#ffd93d', linestyle='--', alpha=0.5, label='~10^15 (supercomputer)')
    
    ax.set_xlabel('Number of Qubits', fontsize=12, color='white')
    ax.set_ylabel('State Space Dimension', fontsize=12, color='white')
    ax.set_title('Exponential Growth of Quantum State Space', fontsize=14, fontweight='bold', color='#64ffda')
    ax.tick_params(colors='white')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('state_space_growth.png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    print("Saved: state_space_growth.png")
    plt.show()


def plot_tensor_product_visual():
    """Visualize tensor product operation."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.patch.set_facecolor('#1a1a2e')
    fig.suptitle('Tensor Product: |ψ₁⟩ ⊗ |ψ₂⟩ = |ψ₁ψ₂⟩', fontsize=14, fontweight='bold', color='white')
    
    # State 1
    ax = axes[0]
    ax.set_facecolor('#0a0a0a')
    state1 = np.array([0.8, 0.6])  # Example state
    ax.bar(['|0⟩', '|1⟩'], np.abs(state1)**2, color=['#4ecdc4', '#4ecdc4'], edgecolor='white')
    ax.set_title('|ψ₁⟩', color='#64ffda', fontsize=12)
    ax.set_ylim(0, 1)
    ax.tick_params(colors='white')
    
    # Tensor symbol
    ax = axes[1]
    ax.set_facecolor('#1a1a2e')
    ax.text(0.5, 0.5, '⊗', fontsize=60, ha='center', va='center', color='#ffd93d')
    ax.axis('off')
    
    # State 2
    ax = axes[2]
    ax.set_facecolor('#0a0a0a')
    state2 = np.array([0.6, 0.8])  # Example state
    ax.bar(['|0⟩', '|1⟩'], np.abs(state2)**2, color=['#ff6b6b', '#ff6b6b'], edgecolor='white')
    ax.set_title('|ψ₂⟩', color='#64ffda', fontsize=12)
    ax.set_ylim(0, 1)
    ax.tick_params(colors='white')
    
    # Result
    ax = axes[3]
    ax.set_facecolor('#0a0a0a')
    result = np.kron(state1, state2)
    ax.bar(['|00⟩', '|01⟩', '|10⟩', '|11⟩'], np.abs(result)**2, 
           color=['#64ffda', '#64ffda', '#64ffda', '#64ffda'], edgecolor='white')
    ax.set_title('|ψ₁⟩ ⊗ |ψ₂⟩', color='#64ffda', fontsize=12)
    ax.set_ylim(0, 1)
    ax.tick_params(colors='white')
    
    plt.tight_layout()
    plt.savefig('tensor_product_visual.png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    print("Saved: tensor_product_visual.png")
    plt.show()


def main():
    """Main function."""
    print("\n" + "=" * 60)
    print("MULTIPLE-QUBIT GATES")
    print("Tensor Products and Multi-Qubit State Spaces")
    print("=" * 60)
    
    demo_tensor_product()
    demo_gate_tensor()
    demo_product_vs_entangled()
    demo_separability()
    demo_state_space_growth()
    
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    plot_two_qubit_states()
    plot_state_space_growth()
    plot_tensor_product_visual()
    
    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("""
1. Tensor product (⊗) combines qubit states
2. n qubits → 2ⁿ dimensional state space
3. Product states: |ψ⟩ = |ψ₁⟩ ⊗ |ψ₂⟩ (separable)
4. Entangled states: CANNOT be written as product
5. Bell states are maximally entangled
6. Multi-qubit gates are 2ⁿ × 2ⁿ unitary matrices
7. Single-qubit gate on qubit k: I⊗...⊗U⊗...⊗I
8. Exponential growth enables quantum advantage
    """)


if __name__ == "__main__":
    main()
