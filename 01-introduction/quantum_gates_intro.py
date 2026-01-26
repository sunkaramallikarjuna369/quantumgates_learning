"""
Introduction to Quantum Gates - Interactive Python Demonstrations
================================================================

This script provides comprehensive demonstrations of quantum gate concepts including:
- Unitary matrix properties
- Gate action on qubit states
- Probability preservation
- Visual comparison of classical vs quantum gates

Run this script to explore quantum gates interactively!

Requirements: numpy, matplotlib
Install: pip install numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Basis states
ket_0 = np.array([[1], [0]], dtype=complex)
ket_1 = np.array([[0], [1]], dtype=complex)

# Common quantum gates
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)


def is_unitary(matrix, tolerance=1e-10):
    """
    Check if a matrix is unitary (U†U = I).
    
    Parameters:
    -----------
    matrix : numpy.ndarray
        Matrix to check
    tolerance : float
        Numerical tolerance for comparison
    
    Returns:
    --------
    bool
        True if matrix is unitary
    """
    n = matrix.shape[0]
    product = np.dot(matrix.conj().T, matrix)
    identity = np.eye(n)
    return np.allclose(product, identity, atol=tolerance)


def create_qubit_state(alpha, beta):
    """
    Create a normalized qubit state |ψ⟩ = α|0⟩ + β|1⟩
    
    Parameters:
    -----------
    alpha : complex
        Amplitude for |0⟩ state
    beta : complex
        Amplitude for |1⟩ state
    
    Returns:
    --------
    numpy.ndarray
        Normalized qubit state vector
    """
    state = alpha * ket_0 + beta * ket_1
    norm = np.sqrt(np.abs(alpha)**2 + np.abs(beta)**2)
    return state / norm


def apply_gate(gate, state):
    """
    Apply a quantum gate to a state.
    
    Parameters:
    -----------
    gate : numpy.ndarray
        Unitary gate matrix
    state : numpy.ndarray
        Qubit state vector
    
    Returns:
    --------
    numpy.ndarray
        Transformed state
    """
    return np.dot(gate, state)


def get_probabilities(state):
    """
    Get measurement probabilities from a state.
    
    Parameters:
    -----------
    state : numpy.ndarray
        Qubit state vector
    
    Returns:
    --------
    tuple
        (probability of |0⟩, probability of |1⟩)
    """
    prob_0 = np.abs(state[0, 0])**2
    prob_1 = np.abs(state[1, 0])**2
    return prob_0, prob_1


def state_to_bloch(state):
    """
    Convert a qubit state to Bloch sphere coordinates.
    
    Parameters:
    -----------
    state : numpy.ndarray
        Qubit state vector
    
    Returns:
    --------
    tuple
        (x, y, z) Bloch coordinates
    """
    alpha = state[0, 0]
    beta = state[1, 0]
    
    # Calculate Bloch coordinates
    x = 2 * np.real(alpha * np.conj(beta))
    y = 2 * np.imag(alpha * np.conj(beta))
    z = np.abs(alpha)**2 - np.abs(beta)**2
    
    return x, y, z


def demo_unitarity():
    """Demonstrate that quantum gates are unitary."""
    print("=" * 60)
    print("DEMONSTRATION: Unitarity of Quantum Gates")
    print("=" * 60)
    print("\nA matrix U is unitary if U†U = UU† = I")
    print("This ensures probability preservation.\n")
    
    gates = {
        'Identity (I)': I,
        'Pauli-X': X,
        'Pauli-Y': Y,
        'Pauli-Z': Z,
        'Hadamard (H)': H,
        'Phase (S)': S,
        'T Gate': T
    }
    
    for name, gate in gates.items():
        unitary = is_unitary(gate)
        status = "UNITARY" if unitary else "NOT UNITARY"
        print(f"{name:15} : {status}")
        
        # Show U†U = I
        product = np.dot(gate.conj().T, gate)
        print(f"  U†U = \n{np.round(product, 3)}\n")


def demo_probability_preservation():
    """Demonstrate that gates preserve probability."""
    print("=" * 60)
    print("DEMONSTRATION: Probability Preservation")
    print("=" * 60)
    print("\nFor any state |ψ⟩ = α|0⟩ + β|1⟩, we have |α|² + |β|² = 1")
    print("This must remain true after applying any gate.\n")
    
    # Create a random state
    alpha = 0.6 + 0.2j
    beta = 0.5 - 0.3j
    state = create_qubit_state(alpha, beta)
    
    prob_0, prob_1 = get_probabilities(state)
    print(f"Initial state: α={state[0,0]:.4f}, β={state[1,0]:.4f}")
    print(f"Initial probabilities: P(0)={prob_0:.6f}, P(1)={prob_1:.6f}")
    print(f"Sum = {prob_0 + prob_1:.6f}\n")
    
    gates = {'X': X, 'Y': Y, 'Z': Z, 'H': H, 'S': S, 'T': T}
    
    for name, gate in gates.items():
        new_state = apply_gate(gate, state)
        new_prob_0, new_prob_1 = get_probabilities(new_state)
        total = new_prob_0 + new_prob_1
        print(f"After {name}: P(0)={new_prob_0:.6f}, P(1)={new_prob_1:.6f}, Sum={total:.6f}")


def demo_gate_action():
    """Demonstrate how gates transform states."""
    print("\n" + "=" * 60)
    print("DEMONSTRATION: Gate Action on States")
    print("=" * 60)
    
    print("\n--- Pauli-X Gate (Bit Flip) ---")
    print("X|0⟩ = |1⟩")
    result = apply_gate(X, ket_0)
    print(f"Result: [{result[0,0]:.0f}, {result[1,0]:.0f}]ᵀ = |1⟩")
    
    print("\nX|1⟩ = |0⟩")
    result = apply_gate(X, ket_1)
    print(f"Result: [{result[0,0]:.0f}, {result[1,0]:.0f}]ᵀ = |0⟩")
    
    print("\n--- Hadamard Gate (Superposition) ---")
    print("H|0⟩ = (|0⟩ + |1⟩)/√2 = |+⟩")
    result = apply_gate(H, ket_0)
    print(f"Result: [{result[0,0]:.4f}, {result[1,0]:.4f}]ᵀ")
    
    print("\nH|1⟩ = (|0⟩ - |1⟩)/√2 = |-⟩")
    result = apply_gate(H, ket_1)
    print(f"Result: [{result[0,0]:.4f}, {result[1,0]:.4f}]ᵀ")
    
    print("\n--- Pauli-Z Gate (Phase Flip) ---")
    print("Z|0⟩ = |0⟩")
    result = apply_gate(Z, ket_0)
    print(f"Result: [{result[0,0]:.0f}, {result[1,0]:.0f}]ᵀ")
    
    print("\nZ|1⟩ = -|1⟩")
    result = apply_gate(Z, ket_1)
    print(f"Result: [{result[0,0]:.0f}, {result[1,0]:.0f}]ᵀ")


def plot_classical_vs_quantum():
    """Visualize classical bits vs quantum qubits."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Classical bit
    ax1.set_xlim(-0.5, 1.5)
    ax1.set_ylim(-0.5, 1.5)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title('Classical Bit\n(Discrete States)', fontsize=16, fontweight='bold', pad=20)
    
    circle0 = plt.Circle((0.2, 0.5), 0.25, color='#ff6b6b', alpha=0.8)
    ax1.add_patch(circle0)
    ax1.text(0.2, 0.5, '0', ha='center', va='center', fontsize=28, fontweight='bold', color='white')
    
    circle1 = plt.Circle((0.8, 0.5), 0.25, color='#4ecdc4', alpha=0.8)
    ax1.add_patch(circle1)
    ax1.text(0.8, 0.5, '1', ha='center', va='center', fontsize=28, fontweight='bold', color='white')
    
    ax1.annotate('', xy=(0.55, 0.5), xytext=(0.45, 0.5),
                arrowprops=dict(arrowstyle='<->', lw=3, color='gray'))
    ax1.text(0.5, 0.75, 'Discrete\nTransition', ha='center', fontsize=12, color='gray')
    
    # Quantum qubit (Bloch sphere)
    ax2 = plt.subplot(122, projection='3d')
    ax2.set_title('Quantum Qubit\n(Continuous States on Bloch Sphere)', fontsize=16, fontweight='bold', pad=20)
    
    # Draw sphere
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax2.plot_surface(x, y, z, alpha=0.1, color='cyan')
    
    # Axes
    ax2.plot([0, 1.3], [0, 0], [0, 0], 'r-', lw=2, label='X')
    ax2.plot([0, 0], [0, 1.3], [0, 0], 'g-', lw=2, label='Y')
    ax2.plot([0, 0], [0, 0], [0, 1.3], 'b-', lw=2, label='Z')
    
    # Basis states
    ax2.scatter([0], [0], [1], color='#ff6b6b', s=200, marker='o', label='|0⟩')
    ax2.scatter([0], [0], [-1], color='#4ecdc4', s=200, marker='o', label='|1⟩')
    
    # Superposition states
    ax2.scatter([1], [0], [0], color='#ffd93d', s=150, marker='o', label='|+⟩')
    ax2.scatter([-1], [0], [0], color='#a8e6cf', s=150, marker='o', label='|-⟩')
    
    # State vector
    theta, phi = np.pi/4, np.pi/3
    sx = np.sin(theta) * np.cos(phi)
    sy = np.sin(theta) * np.sin(phi)
    sz = np.cos(theta)
    ax2.quiver(0, 0, 0, sx, sy, sz, color='purple', arrow_length_ratio=0.15, linewidth=3)
    
    ax2.set_xlabel('X', fontsize=12)
    ax2.set_ylabel('Y', fontsize=12)
    ax2.set_zlabel('Z', fontsize=12)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_box_aspect([1,1,1])
    
    plt.tight_layout()
    plt.savefig('classical_vs_quantum.png', dpi=150, bbox_inches='tight', 
                facecolor='#1a1a2e', edgecolor='none')
    print("\nSaved: classical_vs_quantum.png")
    plt.show()


def plot_gate_matrices():
    """Visualize quantum gate matrices."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Quantum Gate Matrices', fontsize=18, fontweight='bold', color='white')
    fig.patch.set_facecolor('#1a1a2e')
    
    gates = [
        ('Identity (I)', I),
        ('Pauli-X', X),
        ('Pauli-Y', Y),
        ('Pauli-Z', Z),
        ('Hadamard (H)', H),
        ('Phase (S)', S),
        ('T Gate', T),
        ('X² = I', np.dot(X, X))
    ]
    
    for idx, (name, gate) in enumerate(gates):
        ax = axes[idx // 4, idx % 4]
        ax.set_facecolor('#0a0a0a')
        
        # Create matrix visualization
        real_part = np.real(gate)
        imag_part = np.imag(gate)
        
        # Display matrix
        im = ax.imshow(np.abs(gate), cmap='viridis', vmin=0, vmax=1.5)
        
        # Add text annotations
        for i in range(2):
            for j in range(2):
                val = gate[i, j]
                if np.abs(np.imag(val)) < 1e-10:
                    text = f'{np.real(val):.2f}'
                elif np.abs(np.real(val)) < 1e-10:
                    text = f'{np.imag(val):.2f}i'
                else:
                    text = f'{np.real(val):.2f}+{np.imag(val):.2f}i'
                ax.text(j, i, text, ha='center', va='center', 
                       color='white', fontsize=10, fontweight='bold')
        
        ax.set_title(name, fontsize=12, fontweight='bold', color='#64ffda')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['|0⟩', '|1⟩'], color='white')
        ax.set_yticklabels(['⟨0|', '⟨1|'], color='white')
    
    plt.tight_layout()
    plt.savefig('gate_matrices.png', dpi=150, bbox_inches='tight',
                facecolor='#1a1a2e', edgecolor='none')
    print("Saved: gate_matrices.png")
    plt.show()


def plot_gate_action_bloch():
    """Visualize gate actions on the Bloch sphere."""
    fig = plt.figure(figsize=(15, 10))
    fig.patch.set_facecolor('#1a1a2e')
    
    gates = [
        ('X Gate (π rotation about X)', X, 'red'),
        ('Y Gate (π rotation about Y)', Y, 'green'),
        ('Z Gate (π rotation about Z)', Z, 'blue'),
        ('H Gate (X+Z axis rotation)', H, 'purple')
    ]
    
    for idx, (name, gate, color) in enumerate(gates):
        ax = fig.add_subplot(2, 2, idx + 1, projection='3d')
        ax.set_facecolor('#0a0a0a')
        ax.set_title(name, fontsize=12, fontweight='bold', color='#64ffda', pad=10)
        
        # Draw sphere
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 30)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, alpha=0.1, color='cyan')
        
        # Axes
        ax.plot([0, 1.2], [0, 0], [0, 0], 'r-', lw=1, alpha=0.5)
        ax.plot([0, 0], [0, 1.2], [0, 0], 'g-', lw=1, alpha=0.5)
        ax.plot([0, 0], [0, 0], [0, 1.2], 'b-', lw=1, alpha=0.5)
        
        # Initial state |0⟩
        ax.quiver(0, 0, 0, 0, 0, 1, color='#4ecdc4', arrow_length_ratio=0.15, 
                 linewidth=3, label='Initial |0⟩')
        
        # Apply gate and show result
        result = apply_gate(gate, ket_0)
        bx, by, bz = state_to_bloch(result)
        ax.quiver(0, 0, 0, bx, by, bz, color='#ff6b6b', arrow_length_ratio=0.15,
                 linewidth=3, label='After gate')
        
        ax.set_xlabel('X', color='white')
        ax.set_ylabel('Y', color='white')
        ax.set_zlabel('Z', color='white')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_box_aspect([1,1,1])
    
    plt.tight_layout()
    plt.savefig('gate_action_bloch.png', dpi=150, bbox_inches='tight',
                facecolor='#1a1a2e', edgecolor='none')
    print("Saved: gate_action_bloch.png")
    plt.show()


def interactive_gate_explorer():
    """Interactive exploration of gate effects."""
    print("\n" + "=" * 60)
    print("INTERACTIVE GATE EXPLORER")
    print("=" * 60)
    
    print("\nAvailable gates: I, X, Y, Z, H, S, T")
    print("Enter gate names separated by spaces to compose them.")
    print("Example: 'H X H' applies H, then X, then H")
    print("Type 'quit' to exit.\n")
    
    gate_dict = {'I': I, 'X': X, 'Y': Y, 'Z': Z, 'H': H, 'S': S, 'T': T}
    
    while True:
        try:
            user_input = input("Enter gates (or 'quit'): ").strip().upper()
            
            if user_input == 'QUIT':
                print("Goodbye!")
                break
            
            gate_names = user_input.split()
            
            if not gate_names:
                continue
            
            # Start with |0⟩
            state = ket_0.copy()
            print(f"\nStarting state: |0⟩")
            
            # Apply gates sequentially
            for name in gate_names:
                if name not in gate_dict:
                    print(f"Unknown gate: {name}")
                    continue
                
                gate = gate_dict[name]
                state = apply_gate(gate, state)
                prob_0, prob_1 = get_probabilities(state)
                bx, by, bz = state_to_bloch(state)
                
                print(f"After {name}: α={state[0,0]:.4f}, β={state[1,0]:.4f}")
                print(f"  Probabilities: P(0)={prob_0:.4f}, P(1)={prob_1:.4f}")
                print(f"  Bloch coords: ({bx:.4f}, {by:.4f}, {bz:.4f})")
            
            print()
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main function to run all demonstrations."""
    print("\n" + "=" * 60)
    print("QUANTUM GATES - INTRODUCTION")
    print("Interactive Python Demonstrations")
    print("=" * 60)
    
    # Run demonstrations
    demo_unitarity()
    demo_probability_preservation()
    demo_gate_action()
    
    # Generate visualizations
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    plot_classical_vs_quantum()
    plot_gate_matrices()
    plot_gate_action_bloch()
    
    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("""
1. Quantum gates are UNITARY matrices (U†U = I)
2. Unitarity ensures PROBABILITY PRESERVATION
3. Gates transform states: |ψ'⟩ = U|ψ⟩
4. On Bloch sphere, gates are ROTATIONS
5. Quantum gates are always REVERSIBLE (U⁻¹ = U†)
6. Common gates: X (bit flip), Z (phase flip), H (superposition)
    """)
    
    # Optional: Run interactive explorer
    response = input("\nRun interactive gate explorer? (y/n): ")
    if response.lower() == 'y':
        interactive_gate_explorer()


if __name__ == "__main__":
    main()
