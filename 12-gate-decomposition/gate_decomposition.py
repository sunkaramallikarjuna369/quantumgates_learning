"""
Gate Decomposition - Interactive Python Demonstrations
=====================================================

This script demonstrates gate decomposition:
- Euler angle decomposition
- Two-qubit decomposition
- Common gate decompositions
- Decomposition verification

Requirements: numpy, matplotlib
Install: pip install numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
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

CNOT = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
], dtype=complex)


def Rx(theta):
    """Rotation around X-axis."""
    return np.array([
        [np.cos(theta/2), -1j*np.sin(theta/2)],
        [-1j*np.sin(theta/2), np.cos(theta/2)]
    ], dtype=complex)


def Ry(theta):
    """Rotation around Y-axis."""
    return np.array([
        [np.cos(theta/2), -np.sin(theta/2)],
        [np.sin(theta/2), np.cos(theta/2)]
    ], dtype=complex)


def Rz(theta):
    """Rotation around Z-axis."""
    return np.array([
        [np.exp(-1j*theta/2), 0],
        [0, np.exp(1j*theta/2)]
    ], dtype=complex)


def tensor(*args):
    """Compute tensor product of multiple matrices."""
    result = args[0]
    for m in args[1:]:
        result = np.kron(result, m)
    return result


def matrices_equal(A, B, tol=1e-10):
    """Check if two matrices are equal up to global phase."""
    if A.shape != B.shape:
        return False
    
    # Find first non-zero element
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if np.abs(A[i, j]) > tol and np.abs(B[i, j]) > tol:
                phase = A[i, j] / B[i, j]
                return np.allclose(A, phase * B, atol=tol)
    
    return np.allclose(A, B, atol=tol)


def euler_decomposition(U):
    """
    Decompose a single-qubit unitary into Rz(alpha) * Ry(beta) * Rz(gamma).
    Returns (alpha, beta, gamma, global_phase).
    """
    # Extract matrix elements
    a = U[0, 0]
    b = U[0, 1]
    c = U[1, 0]
    d = U[1, 1]
    
    # Compute beta from |a|
    cos_beta_2 = np.abs(a)
    sin_beta_2 = np.abs(c)
    beta = 2 * np.arctan2(sin_beta_2, cos_beta_2)
    
    # Handle special cases
    if np.abs(np.sin(beta/2)) < 1e-10:
        # beta ≈ 0: U ≈ Rz(alpha + gamma)
        alpha_plus_gamma = -np.angle(a) * 2
        return 0, 0, alpha_plus_gamma, 1
    
    if np.abs(np.cos(beta/2)) < 1e-10:
        # beta ≈ π: special case
        alpha_minus_gamma = np.angle(c) * 2
        return alpha_minus_gamma/2, np.pi, -alpha_minus_gamma/2, 1
    
    # General case
    alpha = -np.angle(a) - np.angle(c)
    gamma = -np.angle(a) + np.angle(c)
    
    return alpha, beta, gamma, 1


def demo_euler_decomposition():
    """Demonstrate Euler angle decomposition."""
    print("=" * 60)
    print("EULER ANGLE DECOMPOSITION")
    print("=" * 60)
    
    print("\nAny single-qubit gate U = Rz(α) · Ry(β) · Rz(γ)")
    
    gates = [
        ('Hadamard (H)', H),
        ('Pauli-X', X),
        ('Pauli-Y', Y),
        ('Pauli-Z', Z),
        ('S gate', S),
        ('T gate', T),
    ]
    
    for name, gate in gates:
        alpha, beta, gamma, _ = euler_decomposition(gate)
        
        # Reconstruct
        reconstructed = np.dot(Rz(alpha), np.dot(Ry(beta), Rz(gamma)))
        
        print(f"\n--- {name} ---")
        print(f"α = {alpha:.4f} rad ({np.degrees(alpha):.1f}°)")
        print(f"β = {beta:.4f} rad ({np.degrees(beta):.1f}°)")
        print(f"γ = {gamma:.4f} rad ({np.degrees(gamma):.1f}°)")
        print(f"Reconstruction matches: {matrices_equal(gate, reconstructed)}")


def demo_hadamard_decomposition():
    """Detailed Hadamard decomposition."""
    print("\n" + "=" * 60)
    print("HADAMARD DECOMPOSITION (DETAILED)")
    print("=" * 60)
    
    print("\nH = Rz(π) · Ry(π/2)")
    print("(up to global phase)")
    
    # Decomposition
    rz_pi = Rz(np.pi)
    ry_pi2 = Ry(np.pi / 2)
    
    print(f"\nRz(π) =\n{np.round(rz_pi, 4)}")
    print(f"\nRy(π/2) =\n{np.round(ry_pi2, 4)}")
    
    product = np.dot(rz_pi, ry_pi2)
    print(f"\nRz(π) · Ry(π/2) =\n{np.round(product, 4)}")
    
    print(f"\nH =\n{np.round(H, 4)}")
    
    print(f"\nMatches (up to global phase): {matrices_equal(H, product)}")


def demo_x_from_h_and_z():
    """Demonstrate X = H · Z · H."""
    print("\n" + "=" * 60)
    print("X = H · Z · H")
    print("=" * 60)
    
    print("\nPauli-X can be constructed from H and Z:")
    
    result = np.dot(H, np.dot(Z, H))
    
    print(f"\nH · Z · H =\n{np.round(result, 4)}")
    print(f"\nX =\n{np.round(X, 4)}")
    print(f"\nMatches: {np.allclose(result, X)}")


def demo_swap_decomposition():
    """Demonstrate SWAP = 3 CNOTs."""
    print("\n" + "=" * 60)
    print("SWAP DECOMPOSITION")
    print("=" * 60)
    
    print("\nSWAP = CNOT₁₂ · CNOT₂₁ · CNOT₁₂")
    
    # CNOT with control on qubit 0, target on qubit 1
    CNOT_01 = CNOT
    
    # CNOT with control on qubit 1, target on qubit 0
    CNOT_10 = np.array([
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0]
    ], dtype=complex)
    
    # Compose
    result = np.dot(CNOT_01, np.dot(CNOT_10, CNOT_01))
    
    SWAP = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ], dtype=complex)
    
    print(f"\nCNOT₁₂ · CNOT₂₁ · CNOT₁₂ =\n{result.astype(int)}")
    print(f"\nSWAP =\n{SWAP.astype(int)}")
    print(f"\nMatches: {np.allclose(result, SWAP)}")


def demo_cz_from_cnot():
    """Demonstrate CZ = (I⊗H) · CNOT · (I⊗H)."""
    print("\n" + "=" * 60)
    print("CZ FROM CNOT")
    print("=" * 60)
    
    print("\nCZ = (I⊗H) · CNOT · (I⊗H)")
    
    IH = tensor(I, H)
    
    result = np.dot(IH, np.dot(CNOT, IH))
    
    CZ = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, -1]
    ], dtype=complex)
    
    print(f"\n(I⊗H) · CNOT · (I⊗H) =\n{np.round(result, 4)}")
    print(f"\nCZ =\n{CZ.astype(int)}")
    print(f"\nMatches: {np.allclose(result, CZ)}")


def demo_arbitrary_rotation():
    """Decompose an arbitrary rotation."""
    print("\n" + "=" * 60)
    print("ARBITRARY ROTATION DECOMPOSITION")
    print("=" * 60)
    
    # Create an arbitrary single-qubit unitary
    theta, phi, lam = np.pi/5, np.pi/7, np.pi/11
    
    # U3 gate: general single-qubit unitary
    U = np.array([
        [np.cos(theta/2), -np.exp(1j*lam)*np.sin(theta/2)],
        [np.exp(1j*phi)*np.sin(theta/2), np.exp(1j*(phi+lam))*np.cos(theta/2)]
    ], dtype=complex)
    
    print(f"\nArbitrary unitary U (θ={theta:.3f}, φ={phi:.3f}, λ={lam:.3f}):")
    print(f"{np.round(U, 4)}")
    
    # Decompose
    alpha, beta, gamma, _ = euler_decomposition(U)
    
    print(f"\nEuler decomposition:")
    print(f"α = {alpha:.4f}, β = {beta:.4f}, γ = {gamma:.4f}")
    
    # Reconstruct
    reconstructed = np.dot(Rz(alpha), np.dot(Ry(beta), Rz(gamma)))
    
    print(f"\nReconstructed =\n{np.round(reconstructed, 4)}")
    print(f"\nMatches (up to global phase): {matrices_equal(U, reconstructed)}")


def plot_decomposition_costs():
    """Visualize decomposition costs."""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#0a0a0a')
    
    gates = ['SWAP', 'CZ', 'iSWAP', 'Toffoli', 'Fredkin', 'Arbitrary\n2-qubit']
    cnots = [3, 1, 2, 6, 8, 3]
    
    colors = ['#64ffda', '#ff6b6b', '#ffd93d', '#a78bfa', '#4ecdc4', '#ff8e8e']
    
    bars = ax.bar(gates, cnots, color=colors, edgecolor='white', linewidth=2)
    
    ax.set_ylabel('CNOT Gates Required', color='white', fontsize=12)
    ax.set_title('Gate Decomposition Costs', fontsize=14, fontweight='bold', color='#64ffda')
    ax.tick_params(colors='white')
    
    # Add value labels
    for bar, val in zip(bars, cnots):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
               str(val), ha='center', va='bottom', color='white', fontsize=12, fontweight='bold')
    
    ax.set_ylim(0, 10)
    
    plt.tight_layout()
    plt.savefig('decomposition_costs.png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    print("\nSaved: decomposition_costs.png")
    plt.show()


def plot_euler_angles():
    """Visualize Euler angle decomposition."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('#1a1a2e')
    fig.suptitle('Euler Angle Decomposition: U = Rz(α) · Ry(β) · Rz(γ)', 
                fontsize=14, fontweight='bold', color='white')
    
    gates = [
        ('Hadamard', H, '#64ffda'),
        ('T gate', T, '#ff6b6b'),
        ('S gate', S, '#ffd93d'),
    ]
    
    for ax, (name, gate, color) in zip(axes, gates):
        ax.set_facecolor('#0a0a0a')
        
        alpha, beta, gamma, _ = euler_decomposition(gate)
        angles = [alpha, beta, gamma]
        labels = ['α (Rz)', 'β (Ry)', 'γ (Rz)']
        
        bars = ax.bar(labels, [np.degrees(a) for a in angles], color=color, edgecolor='white')
        ax.set_ylabel('Angle (degrees)', color='white')
        ax.set_title(name, color=color, fontsize=12)
        ax.tick_params(colors='white')
        ax.axhline(y=0, color='white', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, angles):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                   f'{np.degrees(val):.1f}°', ha='center', va='bottom', 
                   color='white', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('euler_angles.png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    print("Saved: euler_angles.png")
    plt.show()


def main():
    """Main function."""
    print("\n" + "=" * 60)
    print("GATE DECOMPOSITION")
    print("Breaking Complex Operations into Simple Gates")
    print("=" * 60)
    
    demo_euler_decomposition()
    demo_hadamard_decomposition()
    demo_x_from_h_and_z()
    demo_swap_decomposition()
    demo_cz_from_cnot()
    demo_arbitrary_rotation()
    
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    plot_decomposition_costs()
    plot_euler_angles()
    
    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("""
1. Any single-qubit gate = Rz(α) · Ry(β) · Rz(γ) (Euler)
2. Any two-qubit gate needs at most 3 CNOTs (KAK)
3. SWAP = 3 CNOTs
4. CZ = (I⊗H) · CNOT · (I⊗H)
5. X = H · Z · H (basis change)
6. Toffoli requires ~6 CNOTs
7. Minimizing CNOT count is crucial for real hardware
8. Gate decomposition is key to quantum compilation
    """)


if __name__ == "__main__":
    main()
