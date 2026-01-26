"""
Single-Qubit Gates - Interactive Python Demonstrations
=====================================================

This script demonstrates single-qubit gates and their properties:
- General U(θ, φ, λ) parameterization
- Bloch sphere rotations
- Gate composition
- Visualization of gate actions

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


def U_gate(theta, phi, lam):
    """
    General single-qubit unitary gate U(θ, φ, λ).
    
    U(θ, φ, λ) = [cos(θ/2)           -e^(iλ)sin(θ/2)    ]
                 [e^(iφ)sin(θ/2)      e^(i(φ+λ))cos(θ/2)]
    
    Parameters:
    -----------
    theta : float
        Rotation angle (0 to π)
    phi : float
        Azimuthal angle
    lam : float
        Phase parameter
    
    Returns:
    --------
    numpy.ndarray
        2x2 unitary matrix
    """
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    
    return np.array([
        [c, -np.exp(1j * lam) * s],
        [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]
    ], dtype=complex)


def Rx(theta):
    """Rotation about X-axis by angle theta."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([
        [c, -1j * s],
        [-1j * s, c]
    ], dtype=complex)


def Ry(theta):
    """Rotation about Y-axis by angle theta."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([
        [c, -s],
        [s, c]
    ], dtype=complex)


def Rz(theta):
    """Rotation about Z-axis by angle theta."""
    return np.array([
        [np.exp(-1j * theta / 2), 0],
        [0, np.exp(1j * theta / 2)]
    ], dtype=complex)


# Standard gates
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)


def state_to_bloch(state):
    """Convert qubit state to Bloch sphere coordinates."""
    alpha = state[0, 0]
    beta = state[1, 0]
    
    x = 2 * np.real(alpha * np.conj(beta))
    y = 2 * np.imag(alpha * np.conj(beta))
    z = np.abs(alpha)**2 - np.abs(beta)**2
    
    return x, y, z


def bloch_to_state(theta, phi):
    """Convert Bloch sphere angles to state vector."""
    alpha = np.cos(theta / 2)
    beta = np.exp(1j * phi) * np.sin(theta / 2)
    return np.array([[alpha], [beta]], dtype=complex)


def apply_gate(gate, state):
    """Apply gate to state."""
    return np.dot(gate, state)


def demo_general_unitary():
    """Demonstrate the general U(θ, φ, λ) parameterization."""
    print("=" * 60)
    print("GENERAL UNITARY U(θ, φ, λ)")
    print("=" * 60)
    
    print("\nU(θ, φ, λ) = [cos(θ/2)           -e^(iλ)sin(θ/2)    ]")
    print("             [e^(iφ)sin(θ/2)      e^(i(φ+λ))cos(θ/2)]")
    
    print("\n--- Special Cases ---")
    
    # X gate: θ=π, φ=0, λ=π
    print("\nPauli-X: U(π, 0, π)")
    U_X = U_gate(np.pi, 0, np.pi)
    print(f"U = \n{np.round(U_X, 4)}")
    print(f"Matches X? {np.allclose(U_X, X)}")
    
    # Y gate: θ=π, φ=π/2, λ=π/2
    print("\nPauli-Y: U(π, π/2, π/2)")
    U_Y = U_gate(np.pi, np.pi/2, np.pi/2)
    print(f"U = \n{np.round(U_Y, 4)}")
    print(f"Matches Y? {np.allclose(U_Y, Y)}")
    
    # Z gate: θ=0, φ=0, λ=π
    print("\nPauli-Z: U(0, 0, π)")
    U_Z = U_gate(0, 0, np.pi)
    print(f"U = \n{np.round(U_Z, 4)}")
    print(f"Matches Z? {np.allclose(U_Z, Z)}")
    
    # H gate: θ=π/2, φ=0, λ=π
    print("\nHadamard: U(π/2, 0, π)")
    U_H = U_gate(np.pi/2, 0, np.pi)
    print(f"U = \n{np.round(U_H, 4)}")
    print(f"Matches H? {np.allclose(U_H, H)}")


def demo_rotation_gates():
    """Demonstrate rotation gates Rx, Ry, Rz."""
    print("\n" + "=" * 60)
    print("ROTATION GATES")
    print("=" * 60)
    
    print("\nRx(θ) = exp(-iθX/2) = cos(θ/2)I - i·sin(θ/2)X")
    print("Ry(θ) = exp(-iθY/2) = cos(θ/2)I - i·sin(θ/2)Y")
    print("Rz(θ) = exp(-iθZ/2) = cos(θ/2)I - i·sin(θ/2)Z")
    
    print("\n--- Pauli gates as π rotations ---")
    
    print("\nX = Rx(π) (up to global phase)")
    Rx_pi = Rx(np.pi)
    print(f"Rx(π) = \n{np.round(Rx_pi, 4)}")
    print(f"Rx(π) = -iX? {np.allclose(Rx_pi, -1j * X)}")
    
    print("\nY = Ry(π) (up to global phase)")
    Ry_pi = Ry(np.pi)
    print(f"Ry(π) = \n{np.round(Ry_pi, 4)}")
    print(f"Ry(π) = -iY? {np.allclose(Ry_pi, -1j * Y)}")
    
    print("\nZ = Rz(π) (up to global phase)")
    Rz_pi = Rz(np.pi)
    print(f"Rz(π) = \n{np.round(Rz_pi, 4)}")
    print(f"Rz(π) = -iZ? {np.allclose(Rz_pi, -1j * Z)}")


def demo_gate_composition():
    """Demonstrate gate composition."""
    print("\n" + "=" * 60)
    print("GATE COMPOSITION")
    print("=" * 60)
    
    print("\nAny single-qubit gate can be decomposed as:")
    print("U = e^(iα) Rz(β) Ry(γ) Rz(δ)")
    
    print("\n--- Example: Decomposing Hadamard ---")
    print("H = Rz(π) Ry(π/2) (up to global phase)")
    
    H_decomp = np.dot(Rz(np.pi), Ry(np.pi/2))
    print(f"\nRz(π)Ry(π/2) = \n{np.round(H_decomp, 4)}")
    print(f"\nH = \n{np.round(H, 4)}")
    
    # Check if equal up to global phase
    ratio = H[0, 0] / H_decomp[0, 0] if H_decomp[0, 0] != 0 else 1
    print(f"\nGlobal phase factor: {ratio:.4f}")
    print(f"Equal up to global phase? {np.allclose(H, ratio * H_decomp)}")
    
    print("\n--- Composing multiple gates ---")
    print("HZH = X")
    HZH = np.dot(H, np.dot(Z, H))
    print(f"HZH = \n{np.round(HZH, 4)}")
    print(f"Equals X? {np.allclose(HZH, X)}")
    
    print("\nHXH = Z")
    HXH = np.dot(H, np.dot(X, H))
    print(f"HXH = \n{np.round(HXH, 4)}")
    print(f"Equals Z? {np.allclose(HXH, Z)}")


def plot_bloch_sphere_rotation(gate, gate_name, initial_state=None):
    """Visualize gate action on Bloch sphere."""
    if initial_state is None:
        initial_state = ket_0
    
    fig = plt.figure(figsize=(10, 8))
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
    
    # Initial state
    ix, iy, iz = state_to_bloch(initial_state)
    ax.quiver(0, 0, 0, ix, iy, iz, color='#4ecdc4', arrow_length_ratio=0.15,
             linewidth=3, label='Initial')
    
    # Apply gate
    final_state = apply_gate(gate, initial_state)
    fx, fy, fz = state_to_bloch(final_state)
    ax.quiver(0, 0, 0, fx, fy, fz, color='#ff6b6b', arrow_length_ratio=0.15,
             linewidth=3, label='Final')
    
    # Draw trajectory (interpolation)
    n_points = 20
    for i in range(n_points):
        t = i / n_points
        # Linear interpolation (simplified)
        px = ix + t * (fx - ix)
        py = iy + t * (fy - iy)
        pz = iz + t * (fz - iz)
        # Normalize to sphere surface
        norm = np.sqrt(px**2 + py**2 + pz**2)
        if norm > 0:
            px, py, pz = px/norm, py/norm, pz/norm
        ax.scatter([px], [py], [pz], color='yellow', s=10, alpha=0.5)
    
    ax.set_xlabel('X', color='white')
    ax.set_ylabel('Y', color='white')
    ax.set_zlabel('Z', color='white')
    ax.set_title(f'{gate_name} Gate Action', fontsize=14, fontweight='bold', color='#64ffda')
    ax.legend(loc='upper right')
    ax.set_box_aspect([1, 1, 1])
    
    plt.tight_layout()
    filename = f'{gate_name.lower()}_gate_action.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    print(f"Saved: {filename}")
    plt.show()


def plot_all_single_qubit_gates():
    """Visualize all common single-qubit gates."""
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor('#1a1a2e')
    
    gates = [
        ('X (Bit Flip)', X),
        ('Y (Bit+Phase Flip)', Y),
        ('Z (Phase Flip)', Z),
        ('H (Hadamard)', H),
        ('S (Phase)', S),
        ('T (π/8)', T)
    ]
    
    for idx, (name, gate) in enumerate(gates):
        ax = fig.add_subplot(2, 3, idx + 1, projection='3d')
        ax.set_facecolor('#0a0a0a')
        
        # Draw sphere
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, alpha=0.1, color='cyan')
        
        # Axes
        ax.plot([0, 1.2], [0, 0], [0, 0], 'r-', lw=1, alpha=0.5)
        ax.plot([0, 0], [0, 1.2], [0, 0], 'g-', lw=1, alpha=0.5)
        ax.plot([0, 0], [0, 0], [0, 1.2], 'b-', lw=1, alpha=0.5)
        
        # Show gate action on |0⟩
        initial = ket_0
        final = apply_gate(gate, initial)
        
        ix, iy, iz = state_to_bloch(initial)
        fx, fy, fz = state_to_bloch(final)
        
        ax.quiver(0, 0, 0, ix, iy, iz, color='#4ecdc4', arrow_length_ratio=0.2, linewidth=2)
        ax.quiver(0, 0, 0, fx, fy, fz, color='#ff6b6b', arrow_length_ratio=0.2, linewidth=2)
        
        ax.set_title(name, fontsize=11, fontweight='bold', color='#64ffda')
        ax.set_box_aspect([1, 1, 1])
    
    plt.suptitle('Single-Qubit Gates: Action on |0⟩', fontsize=16, fontweight='bold', color='white')
    plt.tight_layout()
    plt.savefig('all_single_qubit_gates.png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    print("Saved: all_single_qubit_gates.png")
    plt.show()


def plot_rotation_gates_parametric():
    """Visualize parametric rotation gates."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('#1a1a2e')
    fig.suptitle('Rotation Gates: Effect on |0⟩ Probability', fontsize=14, fontweight='bold', color='white')
    
    thetas = np.linspace(0, 2 * np.pi, 100)
    
    rotations = [
        ('Rx(θ)', Rx, '#ff6b6b'),
        ('Ry(θ)', Ry, '#4ecdc4'),
        ('Rz(θ)', Rz, '#ffd93d')
    ]
    
    for ax, (name, rot_func, color) in zip(axes, rotations):
        ax.set_facecolor('#0a0a0a')
        
        prob_0 = []
        prob_1 = []
        
        for theta in thetas:
            gate = rot_func(theta)
            state = apply_gate(gate, ket_0)
            p0 = np.abs(state[0, 0])**2
            p1 = np.abs(state[1, 0])**2
            prob_0.append(p0)
            prob_1.append(p1)
        
        ax.plot(thetas / np.pi, prob_0, color='#4ecdc4', lw=2, label='P(|0⟩)')
        ax.plot(thetas / np.pi, prob_1, color='#ff6b6b', lw=2, label='P(|1⟩)')
        ax.fill_between(thetas / np.pi, prob_0, alpha=0.3, color='#4ecdc4')
        ax.fill_between(thetas / np.pi, prob_1, alpha=0.3, color='#ff6b6b')
        
        ax.set_xlabel('θ/π', color='white')
        ax.set_ylabel('Probability', color='white')
        ax.set_title(name, fontsize=12, fontweight='bold', color=color)
        ax.legend(loc='upper right')
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.tick_params(colors='white')
    
    plt.tight_layout()
    plt.savefig('rotation_gates_parametric.png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    print("Saved: rotation_gates_parametric.png")
    plt.show()


def interactive_single_qubit_explorer():
    """Interactive exploration of single-qubit gates."""
    print("\n" + "=" * 60)
    print("INTERACTIVE SINGLE-QUBIT GATE EXPLORER")
    print("=" * 60)
    
    print("\nCommands:")
    print("  U theta phi lambda  - Apply U(θ, φ, λ) gate (angles in units of π)")
    print("  Rx theta            - Apply Rx(θ) rotation")
    print("  Ry theta            - Apply Ry(θ) rotation")
    print("  Rz theta            - Apply Rz(θ) rotation")
    print("  X, Y, Z, H, S, T    - Apply standard gate")
    print("  reset               - Reset to |0⟩")
    print("  quit                - Exit")
    
    gate_dict = {'X': X, 'Y': Y, 'Z': Z, 'H': H, 'S': S, 'T': T, 'I': I}
    state = ket_0.copy()
    
    while True:
        try:
            bx, by, bz = state_to_bloch(state)
            p0 = np.abs(state[0, 0])**2
            p1 = np.abs(state[1, 0])**2
            
            print(f"\nCurrent state: α={state[0,0]:.4f}, β={state[1,0]:.4f}")
            print(f"Bloch: ({bx:.4f}, {by:.4f}, {bz:.4f})")
            print(f"Probabilities: P(0)={p0:.4f}, P(1)={p1:.4f}")
            
            user_input = input("\nEnter command: ").strip()
            
            if user_input.lower() == 'quit':
                print("Goodbye!")
                break
            
            if user_input.lower() == 'reset':
                state = ket_0.copy()
                print("Reset to |0⟩")
                continue
            
            parts = user_input.upper().split()
            if not parts:
                continue
            
            cmd = parts[0]
            
            if cmd in gate_dict:
                state = apply_gate(gate_dict[cmd], state)
                print(f"Applied {cmd} gate")
            
            elif cmd == 'U' and len(parts) == 4:
                theta = float(parts[1]) * np.pi
                phi = float(parts[2]) * np.pi
                lam = float(parts[3]) * np.pi
                gate = U_gate(theta, phi, lam)
                state = apply_gate(gate, state)
                print(f"Applied U({parts[1]}π, {parts[2]}π, {parts[3]}π)")
            
            elif cmd == 'RX' and len(parts) == 2:
                theta = float(parts[1]) * np.pi
                state = apply_gate(Rx(theta), state)
                print(f"Applied Rx({parts[1]}π)")
            
            elif cmd == 'RY' and len(parts) == 2:
                theta = float(parts[1]) * np.pi
                state = apply_gate(Ry(theta), state)
                print(f"Applied Ry({parts[1]}π)")
            
            elif cmd == 'RZ' and len(parts) == 2:
                theta = float(parts[1]) * np.pi
                state = apply_gate(Rz(theta), state)
                print(f"Applied Rz({parts[1]}π)")
            
            else:
                print("Unknown command. Try: X, Y, Z, H, S, T, U, Rx, Ry, Rz, reset, quit")
        
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main function."""
    print("\n" + "=" * 60)
    print("SINGLE-QUBIT GATES")
    print("Interactive Python Demonstrations")
    print("=" * 60)
    
    demo_general_unitary()
    demo_rotation_gates()
    demo_gate_composition()
    
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    plot_all_single_qubit_gates()
    plot_rotation_gates_parametric()
    
    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("""
1. Any single-qubit gate is a rotation on the Bloch sphere
2. General form: U(θ, φ, λ) parameterizes all single-qubit unitaries
3. Rotation gates: Rx(θ), Ry(θ), Rz(θ) rotate about respective axes
4. Pauli gates are π rotations: X=Rx(π), Y=Ry(π), Z=Rz(π)
5. Any gate can be decomposed: U = Rz(β)Ry(γ)Rz(δ)
6. Gate composition: HZH=X, HXH=Z (basis change)
    """)
    
    response = input("\nRun interactive explorer? (y/n): ")
    if response.lower() == 'y':
        interactive_single_qubit_explorer()


if __name__ == "__main__":
    main()
