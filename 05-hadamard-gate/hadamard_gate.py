"""
Hadamard Gate - Interactive Python Demonstrations
================================================

This script demonstrates the Hadamard gate and superposition:
- Matrix representation
- Action on basis states
- Creating superposition
- Visualization on Bloch sphere

Requirements: numpy, matplotlib
Install: pip install numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

# Basis states
ket_0 = np.array([[1], [0]], dtype=complex)
ket_1 = np.array([[0], [1]], dtype=complex)

# Gates
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

# Superposition states
ket_plus = (ket_0 + ket_1) / np.sqrt(2)
ket_minus = (ket_0 - ket_1) / np.sqrt(2)


def state_to_bloch(state):
    """Convert qubit state to Bloch sphere coordinates."""
    alpha = state[0, 0]
    beta = state[1, 0]
    x = 2 * np.real(alpha * np.conj(beta))
    y = 2 * np.imag(alpha * np.conj(beta))
    z = np.abs(alpha)**2 - np.abs(beta)**2
    return x, y, z


def apply_gate(gate, state):
    """Apply gate to state."""
    return np.dot(gate, state)


def demo_hadamard_basics():
    """Demonstrate Hadamard gate basics."""
    print("=" * 60)
    print("HADAMARD GATE - THE GATEWAY TO SUPERPOSITION")
    print("=" * 60)
    
    print("\n--- Matrix Representation ---")
    print(f"\nH = 1/√2 * [[1, 1], [1, -1]]")
    print(f"\nH = \n{H}")
    
    print("\n--- Key Properties ---")
    print(f"\n1. Self-inverse: H² = I")
    H_squared = np.dot(H, H)
    print(f"   H² = \n{np.round(H_squared, 4)}")
    print(f"   H² = I? {np.allclose(H_squared, I)}")
    
    print(f"\n2. Hermitian: H† = H")
    print(f"   H† = H? {np.allclose(H.conj().T, H)}")
    
    print(f"\n3. Unitary: H†H = I")
    print(f"   H†H = I? {np.allclose(np.dot(H.conj().T, H), I)}")


def demo_hadamard_action():
    """Demonstrate Hadamard action on states."""
    print("\n" + "=" * 60)
    print("HADAMARD ACTION ON STATES")
    print("=" * 60)
    
    print("\n--- Action on |0⟩ ---")
    H_on_0 = apply_gate(H, ket_0)
    print(f"H|0⟩ = (|0⟩ + |1⟩)/√2 = |+⟩")
    print(f"Result: {H_on_0.flatten()}")
    print(f"Equals |+⟩? {np.allclose(H_on_0, ket_plus)}")
    
    print("\n--- Action on |1⟩ ---")
    H_on_1 = apply_gate(H, ket_1)
    print(f"H|1⟩ = (|0⟩ - |1⟩)/√2 = |-⟩")
    print(f"Result: {H_on_1.flatten()}")
    print(f"Equals |-⟩? {np.allclose(H_on_1, ket_minus)}")
    
    print("\n--- Action on |+⟩ ---")
    H_on_plus = apply_gate(H, ket_plus)
    print(f"H|+⟩ = |0⟩")
    print(f"Result: {H_on_plus.flatten()}")
    print(f"Equals |0⟩? {np.allclose(H_on_plus, ket_0)}")
    
    print("\n--- Action on |-⟩ ---")
    H_on_minus = apply_gate(H, ket_minus)
    print(f"H|-⟩ = |1⟩")
    print(f"Result: {H_on_minus.flatten()}")
    print(f"Equals |1⟩? {np.allclose(H_on_minus, ket_1)}")


def demo_basis_change():
    """Demonstrate Hadamard as basis change."""
    print("\n" + "=" * 60)
    print("HADAMARD AS BASIS CHANGE")
    print("=" * 60)
    
    print("\nHadamard transforms between computational and Hadamard bases:")
    print("  Computational basis: {|0⟩, |1⟩}")
    print("  Hadamard basis: {|+⟩, |-⟩}")
    
    print("\n--- HXH = Z ---")
    HXH = np.dot(H, np.dot(X, H))
    print(f"HXH = \n{np.round(HXH, 4)}")
    print(f"Z = \n{Z}")
    print(f"HXH = Z? {np.allclose(HXH, Z)}")
    
    print("\n--- HZH = X ---")
    HZH = np.dot(H, np.dot(Z, H))
    print(f"HZH = \n{np.round(HZH, 4)}")
    print(f"X = \n{X}")
    print(f"HZH = X? {np.allclose(HZH, X)}")
    
    print("\nThis means:")
    print("  - X errors in computational basis ↔ Z errors in Hadamard basis")
    print("  - Z errors in computational basis ↔ X errors in Hadamard basis")


def demo_superposition_measurement():
    """Demonstrate measurement of superposition states."""
    print("\n" + "=" * 60)
    print("SUPERPOSITION AND MEASUREMENT")
    print("=" * 60)
    
    print("\n--- Measurement Probabilities ---")
    
    states = [
        ('|0⟩', ket_0),
        ('|1⟩', ket_1),
        ('|+⟩ = H|0⟩', ket_plus),
        ('|-⟩ = H|1⟩', ket_minus),
    ]
    
    for name, state in states:
        p0 = np.abs(state[0, 0])**2
        p1 = np.abs(state[1, 0])**2
        print(f"\n{name}:")
        print(f"  P(|0⟩) = {p0:.4f} = {p0*100:.1f}%")
        print(f"  P(|1⟩) = {p1:.4f} = {p1*100:.1f}%")
    
    print("\n--- Simulated Measurements ---")
    np.random.seed(42)
    
    for name, state in [('|+⟩', ket_plus), ('|-⟩', ket_minus)]:
        p0 = np.abs(state[0, 0])**2
        measurements = np.random.choice([0, 1], size=1000, p=[p0, 1-p0])
        count_0 = np.sum(measurements == 0)
        count_1 = np.sum(measurements == 1)
        print(f"\n{name} - 1000 measurements:")
        print(f"  |0⟩: {count_0} ({count_0/10:.1f}%)")
        print(f"  |1⟩: {count_1} ({count_1/10:.1f}%)")


def plot_hadamard_transformation():
    """Visualize Hadamard transformation on Bloch sphere."""
    fig = plt.figure(figsize=(16, 6))
    fig.patch.set_facecolor('#1a1a2e')
    
    transformations = [
        ('|0⟩ → H → |+⟩', ket_0, ket_plus),
        ('|1⟩ → H → |-⟩', ket_1, ket_minus),
        ('|+⟩ → H → |0⟩', ket_plus, ket_0),
    ]
    
    for idx, (title, initial, final) in enumerate(transformations):
        ax = fig.add_subplot(1, 3, idx + 1, projection='3d')
        ax.set_facecolor('#0a0a0a')
        
        # Draw sphere
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 30)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, alpha=0.1, color='cyan')
        
        # Axes
        ax.plot([0, 1.3], [0, 0], [0, 0], 'r-', lw=2, alpha=0.5)
        ax.plot([0, 0], [0, 1.3], [0, 0], 'g-', lw=2, alpha=0.5)
        ax.plot([0, 0], [0, 0], [0, 1.3], 'b-', lw=2, alpha=0.5)
        
        # Hadamard axis (X+Z)/sqrt(2)
        h_axis = np.array([1, 0, 1]) / np.sqrt(2)
        ax.quiver(0, 0, 0, h_axis[0]*1.2, h_axis[1]*1.2, h_axis[2]*1.2, 
                 color='purple', arrow_length_ratio=0.1, linewidth=2, alpha=0.7)
        
        # Initial state
        ix, iy, iz = state_to_bloch(initial)
        ax.quiver(0, 0, 0, ix, iy, iz, color='#4ecdc4', arrow_length_ratio=0.15, 
                 linewidth=3, label='Initial')
        
        # Final state
        fx, fy, fz = state_to_bloch(final)
        ax.quiver(0, 0, 0, fx, fy, fz, color='#ff6b6b', arrow_length_ratio=0.15,
                 linewidth=3, label='Final')
        
        # Draw arc showing rotation
        n_points = 30
        for i in range(n_points):
            t = i / n_points
            # Interpolate along great circle (simplified)
            px = ix + t * (fx - ix)
            py = iy + t * (fy - iy)
            pz = iz + t * (fz - iz)
            norm = np.sqrt(px**2 + py**2 + pz**2)
            if norm > 0:
                px, py, pz = px/norm, py/norm, pz/norm
            ax.scatter([px], [py], [pz], color='yellow', s=10, alpha=0.5)
        
        ax.set_xlabel('X', color='white')
        ax.set_ylabel('Y', color='white')
        ax.set_zlabel('Z', color='white')
        ax.set_title(title, fontsize=12, fontweight='bold', color='#64ffda')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_box_aspect([1, 1, 1])
    
    plt.suptitle('Hadamard Gate Transformations', fontsize=16, fontweight='bold', color='white')
    plt.tight_layout()
    plt.savefig('hadamard_transformations.png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    print("\nSaved: hadamard_transformations.png")
    plt.show()


def plot_superposition_probabilities():
    """Visualize superposition state probabilities."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.patch.set_facecolor('#1a1a2e')
    fig.suptitle('Measurement Probabilities', fontsize=16, fontweight='bold', color='white')
    
    states = [
        ('|0⟩ (Computational)', ket_0),
        ('|+⟩ = H|0⟩ (Superposition)', ket_plus),
        ('|1⟩ (Computational)', ket_1),
        ('|-⟩ = H|1⟩ (Superposition)', ket_minus),
    ]
    
    for ax, (name, state) in zip(axes.flat, states):
        ax.set_facecolor('#0a0a0a')
        
        p0 = np.abs(state[0, 0])**2
        p1 = np.abs(state[1, 0])**2
        
        bars = ax.bar(['|0⟩', '|1⟩'], [p0, p1], color=['#4ecdc4', '#ff6b6b'], 
                     edgecolor='white', linewidth=2)
        
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Probability', color='white')
        ax.set_title(name, fontsize=11, fontweight='bold', color='#64ffda')
        ax.tick_params(colors='white')
        
        for bar, prob in zip(bars, [p0, p1]):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{prob*100:.0f}%', ha='center', va='bottom', 
                   fontsize=14, fontweight='bold', color='white')
        
        ax.axhline(y=0.5, color='yellow', linestyle='--', alpha=0.5, label='50%')
        ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('superposition_probabilities.png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    print("Saved: superposition_probabilities.png")
    plt.show()


def plot_quantum_parallelism():
    """Visualize quantum parallelism with multiple qubits."""
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#0a0a0a')
    
    n_qubits = np.arange(1, 31)
    n_states = 2 ** n_qubits
    
    ax.semilogy(n_qubits, n_states, 'o-', color='#64ffda', linewidth=2, markersize=8)
    ax.fill_between(n_qubits, 1, n_states, alpha=0.3, color='#64ffda')
    
    # Highlight key points
    highlights = [(10, '1,024'), (20, '~1 million'), (30, '~1 billion')]
    for n, label in highlights:
        ax.axvline(x=n, color='#ff6b6b', linestyle='--', alpha=0.5)
        ax.text(n, 2**n * 2, label, ha='center', fontsize=10, color='white')
    
    ax.set_xlabel('Number of Qubits', fontsize=12, color='white')
    ax.set_ylabel('Number of Simultaneous States', fontsize=12, color='white')
    ax.set_title('Quantum Parallelism: Exponential State Space', fontsize=14, fontweight='bold', color='#64ffda')
    ax.tick_params(colors='white')
    ax.grid(alpha=0.3)
    
    # Add annotation
    ax.text(15, 1e5, 'Each qubit in superposition\ndoubles the state space!', 
           fontsize=11, ha='center', color='#ffd93d',
           bbox=dict(boxstyle='round', facecolor='#1a1a2e', edgecolor='#ffd93d'))
    
    plt.tight_layout()
    plt.savefig('quantum_parallelism.png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    print("Saved: quantum_parallelism.png")
    plt.show()


def main():
    """Main function."""
    print("\n" + "=" * 60)
    print("HADAMARD GATE")
    print("The Gateway to Superposition")
    print("=" * 60)
    
    demo_hadamard_basics()
    demo_hadamard_action()
    demo_basis_change()
    demo_superposition_measurement()
    
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    plot_hadamard_transformation()
    plot_superposition_probabilities()
    plot_quantum_parallelism()
    
    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("""
1. Hadamard creates SUPERPOSITION from basis states
2. H|0⟩ = |+⟩ = (|0⟩ + |1⟩)/√2 (equal superposition)
3. H|1⟩ = |-⟩ = (|0⟩ - |1⟩)/√2 (equal superposition, opposite phase)
4. H² = I (self-inverse)
5. HXH = Z, HZH = X (basis change)
6. Superposition enables QUANTUM PARALLELISM
7. n qubits in superposition = 2ⁿ simultaneous states
    """)


if __name__ == "__main__":
    main()
