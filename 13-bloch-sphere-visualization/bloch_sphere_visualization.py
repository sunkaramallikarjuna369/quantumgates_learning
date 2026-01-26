"""
Bloch Sphere Visualization - Interactive Python Demonstrations
=============================================================

This script demonstrates Bloch sphere visualization:
- State representation on the sphere
- Gate actions as rotations
- Special states and their positions
- Animated gate transformations

Requirements: numpy, matplotlib
Install: pip install numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

# Basic gates
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)


def state_to_bloch(state):
    """Convert a quantum state to Bloch sphere coordinates."""
    alpha = state[0, 0]
    beta = state[1, 0]
    
    # Bloch coordinates
    x = 2 * np.real(np.conj(alpha) * beta)
    y = 2 * np.imag(np.conj(alpha) * beta)
    z = np.abs(alpha)**2 - np.abs(beta)**2
    
    return x, y, z


def bloch_to_state(theta, phi):
    """Convert Bloch sphere angles to quantum state."""
    alpha = np.cos(theta / 2)
    beta = np.exp(1j * phi) * np.sin(theta / 2)
    return np.array([[alpha], [beta]], dtype=complex)


def demo_special_states():
    """Demonstrate special states on the Bloch sphere."""
    print("=" * 60)
    print("SPECIAL STATES ON THE BLOCH SPHERE")
    print("=" * 60)
    
    ket_0 = np.array([[1], [0]], dtype=complex)
    ket_1 = np.array([[0], [1]], dtype=complex)
    ket_plus = np.array([[1], [1]], dtype=complex) / np.sqrt(2)
    ket_minus = np.array([[1], [-1]], dtype=complex) / np.sqrt(2)
    ket_plus_i = np.array([[1], [1j]], dtype=complex) / np.sqrt(2)
    ket_minus_i = np.array([[1], [-1j]], dtype=complex) / np.sqrt(2)
    
    states = [
        ('|0⟩', ket_0, '+Z (North pole)'),
        ('|1⟩', ket_1, '-Z (South pole)'),
        ('|+⟩', ket_plus, '+X axis'),
        ('|-⟩', ket_minus, '-X axis'),
        ('|+i⟩', ket_plus_i, '+Y axis'),
        ('|-i⟩', ket_minus_i, '-Y axis'),
    ]
    
    print("\n--- Bloch Coordinates ---")
    for name, state, position in states:
        x, y, z = state_to_bloch(state)
        print(f"{name:6s}: (x={x:6.3f}, y={y:6.3f}, z={z:6.3f}) - {position}")


def demo_gate_rotations():
    """Demonstrate gates as rotations on the Bloch sphere."""
    print("\n" + "=" * 60)
    print("GATES AS ROTATIONS")
    print("=" * 60)
    
    ket_0 = np.array([[1], [0]], dtype=complex)
    
    gates = [
        ('X', X, 'π rotation around X-axis'),
        ('Y', Y, 'π rotation around Y-axis'),
        ('Z', Z, 'π rotation around Z-axis'),
        ('H', H, 'π rotation around (X+Z)/√2'),
        ('S', S, 'π/2 rotation around Z-axis'),
        ('T', T, 'π/4 rotation around Z-axis'),
    ]
    
    print("\n--- Starting from |0⟩ ---")
    x0, y0, z0 = state_to_bloch(ket_0)
    print(f"Initial: (x={x0:.3f}, y={y0:.3f}, z={z0:.3f})")
    
    for name, gate, description in gates:
        new_state = np.dot(gate, ket_0)
        x, y, z = state_to_bloch(new_state)
        print(f"\nAfter {name}: (x={x:6.3f}, y={y:6.3f}, z={z:6.3f})")
        print(f"  {description}")


def demo_rotation_gates():
    """Demonstrate parametric rotation gates."""
    print("\n" + "=" * 60)
    print("PARAMETRIC ROTATIONS")
    print("=" * 60)
    
    def Rx(theta):
        return np.array([
            [np.cos(theta/2), -1j*np.sin(theta/2)],
            [-1j*np.sin(theta/2), np.cos(theta/2)]
        ], dtype=complex)
    
    def Ry(theta):
        return np.array([
            [np.cos(theta/2), -np.sin(theta/2)],
            [np.sin(theta/2), np.cos(theta/2)]
        ], dtype=complex)
    
    def Rz(theta):
        return np.array([
            [np.exp(-1j*theta/2), 0],
            [0, np.exp(1j*theta/2)]
        ], dtype=complex)
    
    ket_0 = np.array([[1], [0]], dtype=complex)
    
    print("\n--- Rx rotations from |0⟩ ---")
    for angle in [0, np.pi/4, np.pi/2, np.pi]:
        state = np.dot(Rx(angle), ket_0)
        x, y, z = state_to_bloch(state)
        print(f"Rx({angle:.2f}): (x={x:6.3f}, y={y:6.3f}, z={z:6.3f})")
    
    print("\n--- Ry rotations from |0⟩ ---")
    for angle in [0, np.pi/4, np.pi/2, np.pi]:
        state = np.dot(Ry(angle), ket_0)
        x, y, z = state_to_bloch(state)
        print(f"Ry({angle:.2f}): (x={x:6.3f}, y={y:6.3f}, z={z:6.3f})")
    
    print("\n--- Rz rotations from |+⟩ ---")
    ket_plus = np.array([[1], [1]], dtype=complex) / np.sqrt(2)
    for angle in [0, np.pi/4, np.pi/2, np.pi]:
        state = np.dot(Rz(angle), ket_plus)
        x, y, z = state_to_bloch(state)
        print(f"Rz({angle:.2f}): (x={x:6.3f}, y={y:6.3f}, z={z:6.3f})")


def plot_bloch_sphere():
    """Create a 3D Bloch sphere visualization."""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#0a0a0a')
    
    # Draw sphere wireframe
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(x, y, z, color='#64ffda', alpha=0.1, linewidth=0.5)
    
    # Draw axes
    ax.quiver(0, 0, 0, 1.5, 0, 0, color='#ff6b6b', arrow_length_ratio=0.1, linewidth=2)
    ax.quiver(0, 0, 0, 0, 1.5, 0, color='#4ecdc4', arrow_length_ratio=0.1, linewidth=2)
    ax.quiver(0, 0, 0, 0, 0, 1.5, color='#ffd93d', arrow_length_ratio=0.1, linewidth=2)
    
    # Labels
    ax.text(1.7, 0, 0, 'X', color='#ff6b6b', fontsize=14, fontweight='bold')
    ax.text(0, 1.7, 0, 'Y', color='#4ecdc4', fontsize=14, fontweight='bold')
    ax.text(0, 0, 1.5, '|0⟩', color='#ffd93d', fontsize=14, fontweight='bold')
    ax.text(0, 0, -1.5, '|1⟩', color='#ffd93d', fontsize=14, fontweight='bold')
    
    # Plot special states
    states = [
        ('|0⟩', 0, 0, 1, '#ffd93d'),
        ('|1⟩', 0, 0, -1, '#ffd93d'),
        ('|+⟩', 1, 0, 0, '#ff6b6b'),
        ('|-⟩', -1, 0, 0, '#ff6b6b'),
        ('|+i⟩', 0, 1, 0, '#4ecdc4'),
        ('|-i⟩', 0, -1, 0, '#4ecdc4'),
    ]
    
    for name, x, y, z, color in states:
        ax.scatter([x], [y], [z], s=100, c=color, marker='o')
    
    # Draw equator
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), np.zeros_like(theta), 
            color='white', alpha=0.3, linewidth=1)
    
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])
    ax.set_xlabel('X', color='white')
    ax.set_ylabel('Y', color='white')
    ax.set_zlabel('Z', color='white')
    ax.tick_params(colors='white')
    ax.set_title('Bloch Sphere with Special States', 
                fontsize=14, fontweight='bold', color='#64ffda')
    
    plt.tight_layout()
    plt.savefig('bloch_sphere.png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    print("\nSaved: bloch_sphere.png")
    plt.show()


def plot_gate_trajectories():
    """Plot trajectories of gates on the Bloch sphere."""
    fig = plt.figure(figsize=(15, 5))
    fig.patch.set_facecolor('#1a1a2e')
    
    def Rx(theta):
        return np.array([
            [np.cos(theta/2), -1j*np.sin(theta/2)],
            [-1j*np.sin(theta/2), np.cos(theta/2)]
        ], dtype=complex)
    
    def Ry(theta):
        return np.array([
            [np.cos(theta/2), -np.sin(theta/2)],
            [np.sin(theta/2), np.cos(theta/2)]
        ], dtype=complex)
    
    def Rz(theta):
        return np.array([
            [np.exp(-1j*theta/2), 0],
            [0, np.exp(1j*theta/2)]
        ], dtype=complex)
    
    rotations = [
        ('Rx rotation', Rx, np.array([[1], [0]], dtype=complex)),
        ('Ry rotation', Ry, np.array([[1], [0]], dtype=complex)),
        ('Rz rotation', Rz, np.array([[1], [1]], dtype=complex) / np.sqrt(2)),
    ]
    
    for idx, (title, rot_func, initial_state) in enumerate(rotations):
        ax = fig.add_subplot(1, 3, idx + 1, projection='3d')
        ax.set_facecolor('#0a0a0a')
        
        # Draw sphere wireframe
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 30)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_wireframe(x, y, z, color='#64ffda', alpha=0.1, linewidth=0.3)
        
        # Draw trajectory
        angles = np.linspace(0, 2*np.pi, 100)
        xs, ys, zs = [], [], []
        for angle in angles:
            state = np.dot(rot_func(angle), initial_state)
            bx, by, bz = state_to_bloch(state)
            xs.append(bx)
            ys.append(by)
            zs.append(bz)
        
        ax.plot(xs, ys, zs, color='#ff6b6b', linewidth=2)
        
        # Mark start and end
        ax.scatter([xs[0]], [ys[0]], [zs[0]], s=100, c='#4ecdc4', marker='o', label='Start')
        
        ax.set_xlim([-1.2, 1.2])
        ax.set_ylim([-1.2, 1.2])
        ax.set_zlim([-1.2, 1.2])
        ax.set_title(title, color='#64ffda', fontsize=12)
        ax.tick_params(colors='white', labelsize=8)
    
    plt.suptitle('Gate Trajectories on Bloch Sphere', 
                fontsize=14, fontweight='bold', color='white')
    plt.tight_layout()
    plt.savefig('gate_trajectories.png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    print("Saved: gate_trajectories.png")
    plt.show()


def main():
    """Main function."""
    print("\n" + "=" * 60)
    print("BLOCH SPHERE VISUALIZATION")
    print("Geometric View of Single-Qubit States and Gates")
    print("=" * 60)
    
    demo_special_states()
    demo_gate_rotations()
    demo_rotation_gates()
    
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    plot_bloch_sphere()
    plot_gate_trajectories()
    
    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("""
1. Any pure qubit state is a point on the Bloch sphere
2. |0⟩ and |1⟩ are at the poles (Z-axis)
3. |+⟩ and |-⟩ are on the X-axis
4. |+i⟩ and |-i⟩ are on the Y-axis
5. Antipodal points are orthogonal states
6. Gates are rotations around axes
7. X, Y, Z are π rotations around their axes
8. S and T are Z-rotations by π/2 and π/4
    """)


if __name__ == "__main__":
    main()
