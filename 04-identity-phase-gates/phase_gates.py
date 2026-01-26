"""
Identity & Phase Gates - Interactive Python Demonstrations
=========================================================

This script demonstrates Identity and Phase gates (I, S, T):
- Matrix representations
- Phase relationships
- Bloch sphere rotations
- Universal computation importance

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
ket_plus = (ket_0 + ket_1) / np.sqrt(2)
ket_minus = (ket_0 - ket_1) / np.sqrt(2)

# Gates
I = np.eye(2, dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
S_dag = np.array([[1, 0], [0, -1j]], dtype=complex)
T_dag = np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=complex)


def Rz(theta):
    """Z-axis rotation gate."""
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


def demo_phase_gates():
    """Demonstrate phase gate properties."""
    print("=" * 60)
    print("IDENTITY & PHASE GATES")
    print("=" * 60)
    
    print("\n--- Matrix Representations ---")
    print(f"\nIdentity (I):\n{I}")
    print(f"\nPhase (S):\n{S}")
    print(f"\nT Gate:\n{T}")
    print(f"\nZ Gate (for comparison):\n{Z}")
    
    print("\n--- Phase Values ---")
    print(f"S adds phase: π/2 = {np.angle(S[1,1]):.4f} rad = {np.degrees(np.angle(S[1,1])):.1f}°")
    print(f"T adds phase: π/4 = {np.angle(T[1,1]):.4f} rad = {np.degrees(np.angle(T[1,1])):.1f}°")
    print(f"Z adds phase: π = {np.angle(Z[1,1]):.4f} rad = {np.degrees(np.angle(Z[1,1])):.1f}°")


def demo_phase_relationships():
    """Demonstrate relationships between phase gates."""
    print("\n" + "=" * 60)
    print("PHASE GATE RELATIONSHIPS")
    print("=" * 60)
    
    print("\n--- T² = S ---")
    T_squared = np.dot(T, T)
    print(f"T² = \n{np.round(T_squared, 4)}")
    print(f"S = \n{S}")
    print(f"T² = S? {np.allclose(T_squared, S)}")
    
    print("\n--- S² = Z ---")
    S_squared = np.dot(S, S)
    print(f"S² = \n{np.round(S_squared, 4)}")
    print(f"Z = \n{Z}")
    print(f"S² = Z? {np.allclose(S_squared, Z)}")
    
    print("\n--- T⁴ = Z ---")
    T_fourth = np.linalg.matrix_power(T, 4)
    print(f"T⁴ = \n{np.round(T_fourth, 4)}")
    print(f"T⁴ = Z? {np.allclose(T_fourth, Z)}")
    
    print("\n--- T⁸ = I ---")
    T_eighth = np.linalg.matrix_power(T, 8)
    print(f"T⁸ = \n{np.round(T_eighth, 4)}")
    print(f"T⁸ = I? {np.allclose(T_eighth, I)}")
    
    print("\n--- Inverse Gates ---")
    print(f"SS† = I? {np.allclose(np.dot(S, S_dag), I)}")
    print(f"TT† = I? {np.allclose(np.dot(T, T_dag), I)}")


def demo_phase_action():
    """Demonstrate phase gate action on states."""
    print("\n" + "=" * 60)
    print("PHASE GATE ACTIONS")
    print("=" * 60)
    
    print("\n--- Action on |0⟩ (no effect) ---")
    print(f"I|0⟩ = |0⟩? {np.allclose(apply_gate(I, ket_0), ket_0)}")
    print(f"S|0⟩ = |0⟩? {np.allclose(apply_gate(S, ket_0), ket_0)}")
    print(f"T|0⟩ = |0⟩? {np.allclose(apply_gate(T, ket_0), ket_0)}")
    
    print("\n--- Action on |1⟩ (phase added) ---")
    S_on_1 = apply_gate(S, ket_1)
    T_on_1 = apply_gate(T, ket_1)
    print(f"S|1⟩ = i|1⟩: {S_on_1.flatten()}")
    print(f"T|1⟩ = e^(iπ/4)|1⟩: {T_on_1.flatten()}")
    
    print("\n--- Action on |+⟩ (rotation on equator) ---")
    S_on_plus = apply_gate(S, ket_plus)
    T_on_plus = apply_gate(T, ket_plus)
    
    print(f"\n|+⟩ = (|0⟩ + |1⟩)/√2")
    print(f"S|+⟩ = (|0⟩ + i|1⟩)/√2 = |+i⟩")
    print(f"Result: {S_on_plus.flatten()}")
    
    print(f"\nT|+⟩ = (|0⟩ + e^(iπ/4)|1⟩)/√2")
    print(f"Result: {T_on_plus.flatten()}")
    
    # Bloch coordinates
    print("\n--- Bloch Sphere Coordinates ---")
    states = [
        ('|+⟩', ket_plus),
        ('T|+⟩', T_on_plus),
        ('S|+⟩', S_on_plus),
        ('Z|+⟩', apply_gate(Z, ket_plus))
    ]
    
    for name, state in states:
        x, y, z = state_to_bloch(state)
        angle = np.degrees(np.arctan2(y, x))
        print(f"{name}: ({x:.4f}, {y:.4f}, {z:.4f}), angle = {angle:.1f}°")


def demo_rz_equivalence():
    """Show phase gates as Rz rotations."""
    print("\n" + "=" * 60)
    print("PHASE GATES AS Rz ROTATIONS")
    print("=" * 60)
    
    print("\nPhase gates are equivalent to Rz rotations (up to global phase):")
    
    print("\n--- Z = Rz(π) ---")
    Rz_pi = Rz(np.pi)
    print(f"Rz(π) = \n{np.round(Rz_pi, 4)}")
    # Z and Rz(π) differ by global phase
    ratio = Z[0,0] / Rz_pi[0,0]
    print(f"Z = {ratio:.4f} * Rz(π)? {np.allclose(Z, ratio * Rz_pi)}")
    
    print("\n--- S = Rz(π/2) ---")
    Rz_pi2 = Rz(np.pi/2)
    print(f"Rz(π/2) = \n{np.round(Rz_pi2, 4)}")
    ratio = S[0,0] / Rz_pi2[0,0]
    print(f"S = {ratio:.4f} * Rz(π/2)? {np.allclose(S, ratio * Rz_pi2)}")
    
    print("\n--- T = Rz(π/4) ---")
    Rz_pi4 = Rz(np.pi/4)
    print(f"Rz(π/4) = \n{np.round(Rz_pi4, 4)}")
    ratio = T[0,0] / Rz_pi4[0,0]
    print(f"T = {ratio:.4f} * Rz(π/4)? {np.allclose(T, ratio * Rz_pi4)}")


def plot_phase_rotation():
    """Visualize phase gate rotations on Bloch sphere equator."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#0a0a0a')
    
    # Draw sphere
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, alpha=0.1, color='cyan')
    
    # Axes
    ax.plot([0, 1.3], [0, 0], [0, 0], 'r-', lw=2, label='X')
    ax.plot([0, 0], [0, 1.3], [0, 0], 'g-', lw=2, label='Y')
    ax.plot([0, 0], [0, 0], [0, 1.3], 'b-', lw=2, label='Z')
    
    # Equator
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), np.zeros_like(theta), 
           'yellow', lw=2, alpha=0.5, label='Equator')
    
    # States showing phase progression
    states = [
        ('|+⟩', ket_plus, '#4ecdc4', 200),
        ('T|+⟩', apply_gate(T, ket_plus), '#a8e6cf', 150),
        ('S|+⟩', apply_gate(S, ket_plus), '#ffd93d', 150),
        ('T³|+⟩', apply_gate(np.linalg.matrix_power(T, 3), ket_plus), '#f39c12', 150),
        ('Z|+⟩ = |-⟩', apply_gate(Z, ket_plus), '#ff6b6b', 200),
    ]
    
    for name, state, color, size in states:
        bx, by, bz = state_to_bloch(state)
        ax.scatter([bx], [by], [bz], color=color, s=size, edgecolors='white', linewidths=2)
        ax.text(bx * 1.2, by * 1.2, bz * 1.2, name, fontsize=10, color='white')
        ax.quiver(0, 0, 0, bx, by, bz, color=color, arrow_length_ratio=0.1, linewidth=2, alpha=0.7)
    
    # Draw arcs showing rotations
    for angle, color, label in [(np.pi/4, '#a8e6cf', 'T'), (np.pi/2, '#ffd93d', 'S'), (np.pi, '#ff6b6b', 'Z')]:
        arc_theta = np.linspace(0, angle, 30)
        ax.plot(0.5 * np.cos(arc_theta), 0.5 * np.sin(arc_theta), np.zeros_like(arc_theta),
               color=color, lw=3, alpha=0.8)
    
    ax.set_xlabel('X', color='white', fontsize=12)
    ax.set_ylabel('Y', color='white', fontsize=12)
    ax.set_zlabel('Z', color='white', fontsize=12)
    ax.set_title('Phase Gates: Rotation on Bloch Sphere Equator', 
                fontsize=14, fontweight='bold', color='#64ffda')
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.savefig('phase_rotation.png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    print("\nSaved: phase_rotation.png")
    plt.show()


def plot_phase_hierarchy():
    """Visualize the hierarchy of phase gates."""
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#0a0a0a')
    
    # Phase angles
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4, 2*np.pi]
    labels = ['I', 'T', 'S=T²', 'T³', 'Z=S²=T⁴', 'T⁵', 'S³=T⁶', 'T⁷', 'I=T⁸']
    colors = ['#4ecdc4', '#a8e6cf', '#ffd93d', '#f39c12', '#ff6b6b', '#f39c12', '#ffd93d', '#a8e6cf', '#4ecdc4']
    
    # Draw unit circle
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'white', lw=1, alpha=0.3)
    
    # Plot phase points
    for angle, label, color in zip(angles[:-1], labels[:-1], colors[:-1]):
        x = np.cos(angle)
        y = np.sin(angle)
        ax.scatter([x], [y], color=color, s=200, edgecolors='white', linewidths=2, zorder=5)
        
        # Label position
        lx = 1.3 * np.cos(angle)
        ly = 1.3 * np.sin(angle)
        ax.text(lx, ly, label, fontsize=11, ha='center', va='center', color='white', fontweight='bold')
    
    # Draw arrows showing T gate progression
    for i in range(len(angles) - 1):
        start_angle = angles[i]
        end_angle = angles[i + 1]
        mid_angle = (start_angle + end_angle) / 2
        
        # Arc
        arc_theta = np.linspace(start_angle, end_angle, 20)
        ax.plot(0.7 * np.cos(arc_theta), 0.7 * np.sin(arc_theta), 
               color='#64ffda', lw=2, alpha=0.5)
    
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Phase Gate Hierarchy: T⁸ = I', fontsize=14, fontweight='bold', color='#64ffda')
    
    # Add legend
    ax.text(0, -1.4, 'Each T gate adds π/4 phase', fontsize=11, ha='center', color='white')
    
    plt.tight_layout()
    plt.savefig('phase_hierarchy.png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    print("Saved: phase_hierarchy.png")
    plt.show()


def main():
    """Main function."""
    print("\n" + "=" * 60)
    print("IDENTITY & PHASE GATES (I, S, T)")
    print("Interactive Python Demonstrations")
    print("=" * 60)
    
    demo_phase_gates()
    demo_phase_relationships()
    demo_phase_action()
    demo_rz_equivalence()
    
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    plot_phase_rotation()
    plot_phase_hierarchy()
    
    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("""
1. Identity (I): Does nothing, used as placeholder
2. Phase (S): Adds π/2 phase to |1⟩, S² = Z
3. T Gate: Adds π/4 phase to |1⟩, T² = S, T⁴ = Z, T⁸ = I
4. Phase gates rotate around Z-axis on Bloch sphere
5. T gate is crucial for universal quantum computation
6. Clifford + T = Universal gate set
7. Phase gates don't change measurement probabilities
    """)


if __name__ == "__main__":
    main()
