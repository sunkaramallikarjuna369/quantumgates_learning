"""
Pauli Gates - Interactive Python Demonstrations
===============================================

This script demonstrates Pauli gates (X, Y, Z) and their properties:
- Matrix representations
- Action on basis states
- Pauli algebra and commutation relations
- Bloch sphere visualization

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

# Pauli matrices
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

# Plus and minus states
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


def demo_pauli_matrices():
    """Demonstrate Pauli matrix properties."""
    print("=" * 60)
    print("PAULI MATRICES")
    print("=" * 60)
    
    print("\n--- Matrix Representations ---")
    print(f"\nPauli-X (Bit Flip):\n{X}")
    print(f"\nPauli-Y (Bit + Phase Flip):\n{Y}")
    print(f"\nPauli-Z (Phase Flip):\n{Z}")
    
    print("\n--- Properties ---")
    print("\n1. Self-inverse (σ² = I):")
    print(f"   X² = I? {np.allclose(np.dot(X, X), I)}")
    print(f"   Y² = I? {np.allclose(np.dot(Y, Y), I)}")
    print(f"   Z² = I? {np.allclose(np.dot(Z, Z), I)}")
    
    print("\n2. Hermitian (σ† = σ):")
    print(f"   X† = X? {np.allclose(X.conj().T, X)}")
    print(f"   Y† = Y? {np.allclose(Y.conj().T, Y)}")
    print(f"   Z† = Z? {np.allclose(Z.conj().T, Z)}")
    
    print("\n3. Unitary (σ†σ = I):")
    print(f"   X†X = I? {np.allclose(np.dot(X.conj().T, X), I)}")
    print(f"   Y†Y = I? {np.allclose(np.dot(Y.conj().T, Y), I)}")
    print(f"   Z†Z = I? {np.allclose(np.dot(Z.conj().T, Z), I)}")
    
    print("\n4. Traceless (Tr(σ) = 0):")
    print(f"   Tr(X) = {np.trace(X)}")
    print(f"   Tr(Y) = {np.trace(Y)}")
    print(f"   Tr(Z) = {np.trace(Z)}")
    
    print("\n5. Determinant = -1:")
    print(f"   det(X) = {np.linalg.det(X):.0f}")
    print(f"   det(Y) = {np.linalg.det(Y):.0f}")
    print(f"   det(Z) = {np.linalg.det(Z):.0f}")


def demo_pauli_action():
    """Demonstrate Pauli gate action on states."""
    print("\n" + "=" * 60)
    print("PAULI GATE ACTIONS")
    print("=" * 60)
    
    print("\n--- Pauli-X (Bit Flip) ---")
    print("X|0⟩ = |1⟩")
    result = apply_gate(X, ket_0)
    print(f"Result: {result.flatten()}")
    print(f"Equals |1⟩? {np.allclose(result, ket_1)}")
    
    print("\nX|1⟩ = |0⟩")
    result = apply_gate(X, ket_1)
    print(f"Result: {result.flatten()}")
    print(f"Equals |0⟩? {np.allclose(result, ket_0)}")
    
    print("\n--- Pauli-Y (Bit + Phase Flip) ---")
    print("Y|0⟩ = i|1⟩")
    result = apply_gate(Y, ket_0)
    print(f"Result: {result.flatten()}")
    print(f"Equals i|1⟩? {np.allclose(result, 1j * ket_1)}")
    
    print("\nY|1⟩ = -i|0⟩")
    result = apply_gate(Y, ket_1)
    print(f"Result: {result.flatten()}")
    print(f"Equals -i|0⟩? {np.allclose(result, -1j * ket_0)}")
    
    print("\n--- Pauli-Z (Phase Flip) ---")
    print("Z|0⟩ = |0⟩")
    result = apply_gate(Z, ket_0)
    print(f"Result: {result.flatten()}")
    print(f"Equals |0⟩? {np.allclose(result, ket_0)}")
    
    print("\nZ|1⟩ = -|1⟩")
    result = apply_gate(Z, ket_1)
    print(f"Result: {result.flatten()}")
    print(f"Equals -|1⟩? {np.allclose(result, -ket_1)}")
    
    print("\n--- Action on Superposition States ---")
    print("\nX|+⟩ = |+⟩ (eigenstate)")
    result = apply_gate(X, ket_plus)
    print(f"Result: {result.flatten()}")
    print(f"Equals |+⟩? {np.allclose(result, ket_plus)}")
    
    print("\nZ|+⟩ = |-⟩")
    result = apply_gate(Z, ket_plus)
    print(f"Result: {result.flatten()}")
    print(f"Equals |-⟩? {np.allclose(result, ket_minus)}")


def demo_pauli_algebra():
    """Demonstrate Pauli algebra relations."""
    print("\n" + "=" * 60)
    print("PAULI ALGEBRA")
    print("=" * 60)
    
    print("\n--- Product Relations ---")
    print("\nXY = iZ")
    XY = np.dot(X, Y)
    print(f"XY = \n{XY}")
    print(f"iZ = \n{1j * Z}")
    print(f"XY = iZ? {np.allclose(XY, 1j * Z)}")
    
    print("\nYZ = iX")
    YZ = np.dot(Y, Z)
    print(f"YZ = iX? {np.allclose(YZ, 1j * X)}")
    
    print("\nZX = iY")
    ZX = np.dot(Z, X)
    print(f"ZX = iY? {np.allclose(ZX, 1j * Y)}")
    
    print("\n--- Anti-commutation ---")
    print("\n{X, Y} = XY + YX = 0")
    anticomm_XY = np.dot(X, Y) + np.dot(Y, X)
    print(f"XY + YX = \n{anticomm_XY}")
    print(f"Equals 0? {np.allclose(anticomm_XY, 0)}")
    
    print("\n--- Commutation ---")
    print("\n[X, Y] = XY - YX = 2iZ")
    comm_XY = np.dot(X, Y) - np.dot(Y, X)
    print(f"XY - YX = \n{comm_XY}")
    print(f"Equals 2iZ? {np.allclose(comm_XY, 2j * Z)}")
    
    print("\n--- Triple Product ---")
    print("\nXYZ = iI")
    XYZ = np.dot(X, np.dot(Y, Z))
    print(f"XYZ = \n{XYZ}")
    print(f"Equals iI? {np.allclose(XYZ, 1j * I)}")
    
    print("\n--- Y in terms of X and Z ---")
    print("\nY = iXZ")
    iXZ = 1j * np.dot(X, Z)
    print(f"iXZ = \n{iXZ}")
    print(f"Equals Y? {np.allclose(iXZ, Y)}")


def demo_eigenvalues():
    """Demonstrate eigenvalues and eigenstates of Pauli matrices."""
    print("\n" + "=" * 60)
    print("EIGENVALUES AND EIGENSTATES")
    print("=" * 60)
    
    paulis = [('X', X), ('Y', Y), ('Z', Z)]
    
    for name, pauli in paulis:
        eigenvalues, eigenvectors = np.linalg.eig(pauli)
        print(f"\n--- Pauli-{name} ---")
        print(f"Eigenvalues: {eigenvalues}")
        print(f"Eigenvectors:")
        for i, (val, vec) in enumerate(zip(eigenvalues, eigenvectors.T)):
            print(f"  λ = {val:+.0f}: {vec}")


def plot_pauli_actions():
    """Visualize Pauli gate actions on Bloch sphere."""
    fig = plt.figure(figsize=(15, 5))
    fig.patch.set_facecolor('#1a1a2e')
    
    gates = [('Pauli-X', X, 'red'), ('Pauli-Y', Y, 'green'), ('Pauli-Z', Z, 'blue')]
    
    for idx, (name, gate, color) in enumerate(gates):
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
        
        # Test states
        test_states = [
            (ket_0, '|0⟩'),
            (ket_1, '|1⟩'),
            (ket_plus, '|+⟩'),
        ]
        
        for state, label in test_states:
            # Initial state
            ix, iy, iz = state_to_bloch(state)
            ax.scatter([ix], [iy], [iz], color='#4ecdc4', s=100, alpha=0.6)
            
            # After gate
            final = apply_gate(gate, state)
            fx, fy, fz = state_to_bloch(final)
            ax.scatter([fx], [fy], [fz], color='#ff6b6b', s=100, alpha=0.6)
            
            # Arrow showing transformation
            ax.quiver(ix, iy, iz, fx-ix, fy-iy, fz-iz, 
                     color='yellow', alpha=0.5, arrow_length_ratio=0.2)
        
        ax.set_xlabel('X', color='white')
        ax.set_ylabel('Y', color='white')
        ax.set_zlabel('Z', color='white')
        ax.set_title(f'{name}\n(π rotation about {color[0].upper()}-axis)', 
                    fontsize=11, fontweight='bold', color='#64ffda')
        ax.set_box_aspect([1, 1, 1])
    
    plt.suptitle('Pauli Gate Actions on Bloch Sphere', fontsize=14, fontweight='bold', color='white')
    plt.tight_layout()
    plt.savefig('pauli_actions.png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    print("\nSaved: pauli_actions.png")
    plt.show()


def plot_pauli_matrices_visual():
    """Visualize Pauli matrices as heatmaps."""
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    fig.patch.set_facecolor('#1a1a2e')
    fig.suptitle('Pauli Matrices Visualization', fontsize=14, fontweight='bold', color='white')
    
    matrices = [('I (Identity)', I), ('X (Bit Flip)', X), ('Y (Bit+Phase)', Y), ('Z (Phase)', Z)]
    
    for ax, (name, matrix) in zip(axes, matrices):
        ax.set_facecolor('#0a0a0a')
        
        # Show real part magnitude
        im = ax.imshow(np.abs(matrix), cmap='viridis', vmin=0, vmax=1)
        
        # Annotate with actual values
        for i in range(2):
            for j in range(2):
                val = matrix[i, j]
                if np.abs(np.imag(val)) < 1e-10:
                    text = f'{np.real(val):.0f}'
                elif np.abs(np.real(val)) < 1e-10:
                    if np.imag(val) == 1:
                        text = 'i'
                    elif np.imag(val) == -1:
                        text = '-i'
                    else:
                        text = f'{np.imag(val):.0f}i'
                else:
                    text = f'{val}'
                ax.text(j, i, text, ha='center', va='center', 
                       color='white', fontsize=16, fontweight='bold')
        
        ax.set_title(name, fontsize=11, fontweight='bold', color='#64ffda')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['|0⟩', '|1⟩'], color='white')
        ax.set_yticklabels(['⟨0|', '⟨1|'], color='white')
    
    plt.tight_layout()
    plt.savefig('pauli_matrices.png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    print("Saved: pauli_matrices.png")
    plt.show()


def interactive_pauli_explorer():
    """Interactive exploration of Pauli gates."""
    print("\n" + "=" * 60)
    print("INTERACTIVE PAULI GATE EXPLORER")
    print("=" * 60)
    
    print("\nCommands: X, Y, Z, I, reset, quit")
    print("Apply gates sequentially to see their effects.")
    
    state = ket_0.copy()
    history = ['|0⟩']
    
    while True:
        try:
            bx, by, bz = state_to_bloch(state)
            p0 = np.abs(state[0, 0])**2
            p1 = np.abs(state[1, 0])**2
            
            print(f"\nState: α={state[0,0]:.4f}, β={state[1,0]:.4f}")
            print(f"Bloch: ({bx:.4f}, {by:.4f}, {bz:.4f})")
            print(f"P(0)={p0:.4f}, P(1)={p1:.4f}")
            print(f"History: {' → '.join(history)}")
            
            cmd = input("\nGate (X/Y/Z/I/reset/quit): ").strip().upper()
            
            if cmd == 'QUIT':
                break
            elif cmd == 'RESET':
                state = ket_0.copy()
                history = ['|0⟩']
            elif cmd in ['X', 'Y', 'Z', 'I']:
                gate = {'X': X, 'Y': Y, 'Z': Z, 'I': I}[cmd]
                state = apply_gate(gate, state)
                history.append(cmd)
            else:
                print("Unknown command")
        
        except KeyboardInterrupt:
            break
    
    print("Goodbye!")


def main():
    """Main function."""
    print("\n" + "=" * 60)
    print("PAULI GATES (X, Y, Z)")
    print("Interactive Python Demonstrations")
    print("=" * 60)
    
    demo_pauli_matrices()
    demo_pauli_action()
    demo_pauli_algebra()
    demo_eigenvalues()
    
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    plot_pauli_matrices_visual()
    plot_pauli_actions()
    
    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("""
1. Pauli-X: Bit flip (quantum NOT), π rotation about X-axis
2. Pauli-Y: Bit + phase flip, π rotation about Y-axis  
3. Pauli-Z: Phase flip, π rotation about Z-axis
4. All Pauli gates are Hermitian, unitary, and self-inverse
5. Pauli algebra: XY=iZ, YZ=iX, ZX=iY, XYZ=iI
6. Anti-commutation: {σᵢ, σⱼ} = 2δᵢⱼI
7. Commutation: [σᵢ, σⱼ] = 2iεᵢⱼₖσₖ
    """)
    
    response = input("\nRun interactive explorer? (y/n): ")
    if response.lower() == 'y':
        interactive_pauli_explorer()


if __name__ == "__main__":
    main()
