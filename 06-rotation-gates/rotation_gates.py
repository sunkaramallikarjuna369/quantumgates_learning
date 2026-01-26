"""
Rotation Gates - Interactive Python Demonstrations
=================================================

This script demonstrates parametric rotation gates Rx, Ry, Rz:
- Matrix representations
- Rotation on Bloch sphere
- Euler decomposition
- Gate composition

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


def Rx(theta):
    """Rotation around X-axis by angle theta."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([
        [c, -1j * s],
        [-1j * s, c]
    ], dtype=complex)


def Ry(theta):
    """Rotation around Y-axis by angle theta."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([
        [c, -s],
        [s, c]
    ], dtype=complex)


def Rz(theta):
    """Rotation around Z-axis by angle theta."""
    return np.array([
        [np.exp(-1j * theta / 2), 0],
        [0, np.exp(1j * theta / 2)]
    ], dtype=complex)


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


def demo_rotation_matrices():
    """Demonstrate rotation gate matrices."""
    print("=" * 60)
    print("ROTATION GATES - Rx, Ry, Rz")
    print("=" * 60)
    
    angles = [0, np.pi/4, np.pi/2, np.pi, 2*np.pi]
    angle_names = ['0', 'π/4', 'π/2', 'π', '2π']
    
    for gate_name, gate_func in [('Rx', Rx), ('Ry', Ry), ('Rz', Rz)]:
        print(f"\n--- {gate_name}(θ) ---")
        for angle, name in zip(angles, angle_names):
            mat = gate_func(angle)
            print(f"\n{gate_name}({name}):")
            print(np.round(mat, 4))


def demo_rotation_properties():
    """Demonstrate rotation gate properties."""
    print("\n" + "=" * 60)
    print("ROTATION GATE PROPERTIES")
    print("=" * 60)
    
    print("\n--- Unitarity ---")
    for name, gate in [('Rx(π/3)', Rx(np.pi/3)), ('Ry(π/4)', Ry(np.pi/4)), ('Rz(π/6)', Rz(np.pi/6))]:
        is_unitary = np.allclose(np.dot(gate.conj().T, gate), I)
        print(f"{name} is unitary: {is_unitary}")
    
    print("\n--- Inverse Property: R(θ)⁻¹ = R(-θ) ---")
    theta = np.pi / 3
    for name, gate in [('Rx', Rx), ('Ry', Ry), ('Rz', Rz)]:
        product = np.dot(gate(theta), gate(-theta))
        is_identity = np.allclose(product, I)
        print(f"{name}(θ) · {name}(-θ) = I: {is_identity}")
    
    print("\n--- Composition: R(α)R(β) = R(α+β) ---")
    alpha, beta = np.pi/4, np.pi/3
    for name, gate in [('Rx', Rx), ('Ry', Ry), ('Rz', Rz)]:
        composed = np.dot(gate(alpha), gate(beta))
        direct = gate(alpha + beta)
        is_equal = np.allclose(composed, direct)
        print(f"{name}(α){name}(β) = {name}(α+β): {is_equal}")
    
    print("\n--- Periodicity: R(θ + 4π) = R(θ) ---")
    theta = np.pi / 5
    for name, gate in [('Rx', Rx), ('Ry', Ry), ('Rz', Rz)]:
        original = gate(theta)
        shifted = gate(theta + 4*np.pi)
        is_equal = np.allclose(original, shifted)
        print(f"{name}(θ + 4π) = {name}(θ): {is_equal}")
    
    print("\n--- Note: R(θ + 2π) = -R(θ) (global phase) ---")
    theta = np.pi / 5
    for name, gate in [('Rx', Rx), ('Ry', Ry), ('Rz', Rz)]:
        original = gate(theta)
        shifted = gate(theta + 2*np.pi)
        is_negative = np.allclose(shifted, -original)
        print(f"{name}(θ + 2π) = -{name}(θ): {is_negative}")


def demo_special_angles():
    """Demonstrate rotation gates at special angles."""
    print("\n" + "=" * 60)
    print("SPECIAL ANGLE VALUES")
    print("=" * 60)
    
    print("\n--- Rx at special angles ---")
    print(f"Rx(0) = I: {np.allclose(Rx(0), I)}")
    print(f"Rx(π) = -iX: {np.allclose(Rx(np.pi), -1j * X)}")
    print(f"Rx(2π) = -I: {np.allclose(Rx(2*np.pi), -I)}")
    
    print("\n--- Ry at special angles ---")
    print(f"Ry(0) = I: {np.allclose(Ry(0), I)}")
    print(f"Ry(π) = -iY: {np.allclose(Ry(np.pi), -1j * Y)}")
    print(f"Ry(2π) = -I: {np.allclose(Ry(2*np.pi), -I)}")
    
    print("\n--- Rz at special angles ---")
    print(f"Rz(0) = I: {np.allclose(Rz(0), I)}")
    print(f"Rz(π) = -iZ: {np.allclose(Rz(np.pi), -1j * Z)}")
    print(f"Rz(2π) = -I: {np.allclose(Rz(2*np.pi), -I)}")
    
    # Phase gates as Rz
    S = np.array([[1, 0], [0, 1j]], dtype=complex)
    T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
    
    print("\n--- Phase gates as Rz (up to global phase) ---")
    # S = Rz(π/2) up to global phase
    rz_pi2 = Rz(np.pi / 2)
    global_phase = rz_pi2[0, 0] / S[0, 0]
    print(f"S = e^(iφ) · Rz(π/2): {np.allclose(S, global_phase * rz_pi2)}")
    
    # T = Rz(π/4) up to global phase
    rz_pi4 = Rz(np.pi / 4)
    global_phase = rz_pi4[0, 0] / T[0, 0]
    print(f"T = e^(iφ) · Rz(π/4): {np.allclose(T, global_phase * rz_pi4)}")


def demo_euler_decomposition():
    """Demonstrate Euler decomposition of arbitrary unitaries."""
    print("\n" + "=" * 60)
    print("EULER DECOMPOSITION")
    print("=" * 60)
    
    print("\nAny single-qubit unitary U can be written as:")
    print("  U = e^(iα) · Rz(β) · Ry(γ) · Rz(δ)")
    print("\nThis is the ZYZ decomposition.")
    
    # Example: Hadamard gate
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    
    print("\n--- Example: Hadamard Gate ---")
    print("H can be decomposed as Rz(π) · Ry(π/2) · Rz(0)")
    
    decomposed = np.dot(Rz(np.pi), np.dot(Ry(np.pi/2), Rz(0)))
    # Check if equal up to global phase
    ratio = H[0, 0] / decomposed[0, 0] if decomposed[0, 0] != 0 else H[1, 0] / decomposed[1, 0]
    is_equal = np.allclose(H, ratio * decomposed)
    print(f"H = e^(iφ) · Rz(π) · Ry(π/2): {is_equal}")
    
    # Alternative: H = Ry(π/2) · Rz(π)
    alt_decomposed = np.dot(Ry(np.pi/2), Rz(np.pi))
    ratio = H[0, 0] / alt_decomposed[0, 0] if alt_decomposed[0, 0] != 0 else H[1, 0] / alt_decomposed[1, 0]
    is_equal = np.allclose(H, ratio * alt_decomposed)
    print(f"H = e^(iφ) · Ry(π/2) · Rz(π): {is_equal}")


def demo_non_commutativity():
    """Demonstrate non-commutativity of rotations."""
    print("\n" + "=" * 60)
    print("NON-COMMUTATIVITY OF ROTATIONS")
    print("=" * 60)
    
    theta = np.pi / 4
    
    print(f"\nFor θ = π/4:")
    
    # Rx and Ry
    RxRy = np.dot(Rx(theta), Ry(theta))
    RyRx = np.dot(Ry(theta), Rx(theta))
    print(f"\nRx(θ) · Ry(θ) = Ry(θ) · Rx(θ)? {np.allclose(RxRy, RyRx)}")
    print("Rx · Ry:")
    print(np.round(RxRy, 4))
    print("Ry · Rx:")
    print(np.round(RyRx, 4))
    
    # Rx and Rz
    RxRz = np.dot(Rx(theta), Rz(theta))
    RzRx = np.dot(Rz(theta), Rx(theta))
    print(f"\nRx(θ) · Rz(θ) = Rz(θ) · Rx(θ)? {np.allclose(RxRz, RzRx)}")
    
    # Ry and Rz
    RyRz = np.dot(Ry(theta), Rz(theta))
    RzRy = np.dot(Rz(theta), Ry(theta))
    print(f"Ry(θ) · Rz(θ) = Rz(θ) · Ry(θ)? {np.allclose(RyRz, RzRy)}")
    
    print("\nRotations around DIFFERENT axes do NOT commute!")
    print("This is fundamental to quantum mechanics and the Bloch sphere geometry.")


def plot_rotation_trajectories():
    """Visualize rotation trajectories on Bloch sphere."""
    fig = plt.figure(figsize=(16, 5))
    fig.patch.set_facecolor('#1a1a2e')
    
    rotations = [
        ('Rx(θ) on |0⟩', Rx, ket_0, '#ff6b6b'),
        ('Ry(θ) on |0⟩', Ry, ket_0, '#4ecdc4'),
        ('Rz(θ) on |+⟩', Rz, (ket_0 + ket_1)/np.sqrt(2), '#ffd93d'),
    ]
    
    for idx, (title, gate_func, initial_state, color) in enumerate(rotations):
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
        ax.text(1.4, 0, 0, 'X', color='red')
        ax.text(0, 1.4, 0, 'Y', color='green')
        ax.text(0, 0, 1.4, 'Z', color='blue')
        
        # Trajectory
        angles = np.linspace(0, 2*np.pi, 100)
        trajectory = []
        for theta in angles:
            state = apply_gate(gate_func(theta), initial_state)
            trajectory.append(state_to_bloch(state))
        
        trajectory = np.array(trajectory)
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
               color=color, linewidth=3, alpha=0.8)
        
        # Initial and final points
        ix, iy, iz = state_to_bloch(initial_state)
        ax.scatter([ix], [iy], [iz], color='white', s=100, marker='o', label='Initial')
        
        # Mark some intermediate points
        for i in [25, 50, 75]:
            ax.scatter([trajectory[i, 0]], [trajectory[i, 1]], [trajectory[i, 2]], 
                      color=color, s=50, alpha=0.7)
        
        ax.set_xlabel('X', color='white')
        ax.set_ylabel('Y', color='white')
        ax.set_zlabel('Z', color='white')
        ax.set_title(title, fontsize=12, fontweight='bold', color='#64ffda')
        ax.set_box_aspect([1, 1, 1])
    
    plt.suptitle('Rotation Gate Trajectories on Bloch Sphere', fontsize=14, fontweight='bold', color='white')
    plt.tight_layout()
    plt.savefig('rotation_trajectories.png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    print("\nSaved: rotation_trajectories.png")
    plt.show()


def plot_rotation_matrices_visual():
    """Visualize rotation matrices as heatmaps."""
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.patch.set_facecolor('#1a1a2e')
    fig.suptitle('Rotation Gate Matrices at Different Angles', fontsize=16, fontweight='bold', color='white')
    
    angles = [0, np.pi/4, np.pi/2, np.pi]
    angle_names = ['0', 'π/4', 'π/2', 'π']
    gates = [('Rx', Rx), ('Ry', Ry), ('Rz', Rz)]
    
    for row, (gate_name, gate_func) in enumerate(gates):
        for col, (angle, angle_name) in enumerate(zip(angles, angle_names)):
            ax = axes[row, col]
            ax.set_facecolor('#0a0a0a')
            
            mat = gate_func(angle)
            
            # Plot magnitude
            magnitude = np.abs(mat)
            im = ax.imshow(magnitude, cmap='viridis', vmin=0, vmax=1)
            
            # Add text annotations
            for i in range(2):
                for j in range(2):
                    val = mat[i, j]
                    text = f'{val.real:.2f}'
                    if abs(val.imag) > 0.01:
                        text += f'\n{val.imag:+.2f}i'
                    ax.text(j, i, text, ha='center', va='center', 
                           color='white', fontsize=10, fontweight='bold')
            
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['0', '1'], color='white')
            ax.set_yticklabels(['0', '1'], color='white')
            ax.set_title(f'{gate_name}({angle_name})', color='#64ffda', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('rotation_matrices.png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    print("Saved: rotation_matrices.png")
    plt.show()


def plot_euler_decomposition():
    """Visualize Euler decomposition."""
    fig = plt.figure(figsize=(14, 5))
    fig.patch.set_facecolor('#1a1a2e')
    
    # Create a random unitary and decompose it
    # For simplicity, use known decomposition of Hadamard
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    
    # H ≈ Rz(π) Ry(π/2) (up to global phase)
    steps = [
        ('Initial |0⟩', ket_0),
        ('After Rz(π)', apply_gate(Rz(np.pi), ket_0)),
        ('After Ry(π/2)', apply_gate(Ry(np.pi/2), apply_gate(Rz(np.pi), ket_0))),
    ]
    
    for idx, (title, state) in enumerate(steps):
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
        
        # State vector
        bx, by, bz = state_to_bloch(state)
        ax.quiver(0, 0, 0, bx, by, bz, color='#ff6b6b', arrow_length_ratio=0.15, linewidth=3)
        
        ax.set_xlabel('X', color='white')
        ax.set_ylabel('Y', color='white')
        ax.set_zlabel('Z', color='white')
        ax.set_title(title, fontsize=12, fontweight='bold', color='#64ffda')
        ax.set_box_aspect([1, 1, 1])
    
    plt.suptitle('Euler Decomposition: H = Ry(π/2) · Rz(π)', fontsize=14, fontweight='bold', color='white')
    plt.tight_layout()
    plt.savefig('euler_decomposition.png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    print("Saved: euler_decomposition.png")
    plt.show()


def main():
    """Main function."""
    print("\n" + "=" * 60)
    print("ROTATION GATES")
    print("Rx, Ry, Rz - Parametric Rotations")
    print("=" * 60)
    
    demo_rotation_matrices()
    demo_rotation_properties()
    demo_special_angles()
    demo_euler_decomposition()
    demo_non_commutativity()
    
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    plot_rotation_trajectories()
    plot_rotation_matrices_visual()
    plot_euler_decomposition()
    
    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("""
1. Rx(θ), Ry(θ), Rz(θ) rotate around X, Y, Z axes
2. R(θ) = exp(-iθσ/2) where σ is the Pauli matrix
3. R(θ)⁻¹ = R(-θ) - inverse is negative angle
4. R(α)R(β) = R(α+β) for same axis
5. Rotations around DIFFERENT axes do NOT commute
6. Period is 4π (not 2π) due to spinor nature
7. Any single-qubit gate = Rz(β)·Ry(γ)·Rz(δ) (Euler)
8. Pauli gates are rotations by π: X=Rx(π), Y=Ry(π), Z=Rz(π)
    """)


if __name__ == "__main__":
    main()
