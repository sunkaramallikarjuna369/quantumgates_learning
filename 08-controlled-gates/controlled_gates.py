"""
Controlled Gates - Interactive Python Demonstrations
===================================================

This script demonstrates controlled gates:
- CNOT (Controlled-X)
- CZ (Controlled-Z)
- Creating entanglement
- Bell states

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

# Single-qubit gates
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

# Two-qubit controlled gates
CNOT = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
], dtype=complex)

CZ = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, -1]
], dtype=complex)


def tensor(a, b):
    """Compute tensor product."""
    return np.kron(a, b)


def demo_cnot_basics():
    """Demonstrate CNOT gate basics."""
    print("=" * 60)
    print("CNOT GATE (Controlled-X)")
    print("=" * 60)
    
    print("\n--- Matrix Representation ---")
    print(f"\nCNOT =\n{CNOT.astype(int)}")
    
    print("\n--- Truth Table ---")
    print(f"{'Input':<10} {'Output':<10} {'Action'}")
    print("-" * 35)
    
    basis_states = [
        (tensor(ket_0, ket_0), '|00⟩'),
        (tensor(ket_0, ket_1), '|01⟩'),
        (tensor(ket_1, ket_0), '|10⟩'),
        (tensor(ket_1, ket_1), '|11⟩'),
    ]
    
    for state, name in basis_states:
        output = np.dot(CNOT, state)
        output_name = None
        for s, n in basis_states:
            if np.allclose(output, s):
                output_name = n
                break
        
        changed = name != output_name
        action = "Target flipped!" if changed else "No change"
        print(f"{name:<10} {output_name:<10} {action}")
    
    print("\n--- Key Properties ---")
    print(f"CNOT² = I: {np.allclose(np.dot(CNOT, CNOT), np.eye(4))}")
    print(f"CNOT is unitary: {np.allclose(np.dot(CNOT.conj().T, CNOT), np.eye(4))}")
    print(f"CNOT is Hermitian: {np.allclose(CNOT, CNOT.conj().T)}")


def demo_cz_gate():
    """Demonstrate CZ gate."""
    print("\n" + "=" * 60)
    print("CZ GATE (Controlled-Z)")
    print("=" * 60)
    
    print("\n--- Matrix Representation ---")
    print(f"\nCZ =\n{CZ.astype(int)}")
    
    print("\n--- Action on Basis States ---")
    basis_states = [
        (tensor(ket_0, ket_0), '|00⟩'),
        (tensor(ket_0, ket_1), '|01⟩'),
        (tensor(ket_1, ket_0), '|10⟩'),
        (tensor(ket_1, ket_1), '|11⟩'),
    ]
    
    for state, name in basis_states:
        output = np.dot(CZ, state)
        # Check if phase changed
        if np.allclose(output, state):
            print(f"CZ{name} = {name}")
        elif np.allclose(output, -state):
            print(f"CZ{name} = -{name}  (phase flip!)")
    
    print("\n--- CZ Symmetry ---")
    print("CZ is symmetric - control and target are interchangeable!")
    print(f"CZ = CZ^T: {np.allclose(CZ, CZ.T)}")


def demo_cnot_cz_relation():
    """Demonstrate relationship between CNOT and CZ."""
    print("\n" + "=" * 60)
    print("CNOT AND CZ RELATIONSHIP")
    print("=" * 60)
    
    print("\nCNOT = (I ⊗ H) · CZ · (I ⊗ H)")
    
    I_H = tensor(I, H)
    converted = np.dot(I_H, np.dot(CZ, I_H))
    
    print(f"\n(I ⊗ H) · CZ · (I ⊗ H) =\n{np.round(converted, 4)}")
    print(f"\nEquals CNOT? {np.allclose(converted, CNOT)}")
    
    print("\nAlternatively: CZ = (I ⊗ H) · CNOT · (I ⊗ H)")
    converted2 = np.dot(I_H, np.dot(CNOT, I_H))
    print(f"Equals CZ? {np.allclose(converted2, CZ)}")


def demo_bell_states():
    """Demonstrate Bell state creation."""
    print("\n" + "=" * 60)
    print("BELL STATES - MAXIMALLY ENTANGLED")
    print("=" * 60)
    
    ket_00 = tensor(ket_0, ket_0)
    ket_01 = tensor(ket_0, ket_1)
    ket_10 = tensor(ket_1, ket_0)
    ket_11 = tensor(ket_1, ket_1)
    
    # Create Bell states using H and CNOT
    H_I = tensor(H, I)
    
    print("\n--- Creating |Φ⁺⟩ from |00⟩ ---")
    print("Step 1: Apply H to first qubit")
    step1 = np.dot(H_I, ket_00)
    print(f"  (H ⊗ I)|00⟩ = {np.round(step1.flatten(), 4)}")
    print("  = (|00⟩ + |10⟩)/√2")
    
    print("\nStep 2: Apply CNOT")
    phi_plus = np.dot(CNOT, step1)
    print(f"  CNOT(|00⟩ + |10⟩)/√2 = {np.round(phi_plus.flatten(), 4)}")
    print("  = (|00⟩ + |11⟩)/√2 = |Φ⁺⟩")
    
    print("\n--- All Four Bell States ---")
    
    # |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
    phi_plus = (ket_00 + ket_11) / np.sqrt(2)
    print(f"\n|Φ⁺⟩ = (|00⟩ + |11⟩)/√2 = {np.round(phi_plus.flatten(), 4)}")
    
    # |Φ⁻⟩ = (|00⟩ - |11⟩)/√2
    phi_minus = (ket_00 - ket_11) / np.sqrt(2)
    print(f"|Φ⁻⟩ = (|00⟩ - |11⟩)/√2 = {np.round(phi_minus.flatten(), 4)}")
    
    # |Ψ⁺⟩ = (|01⟩ + |10⟩)/√2
    psi_plus = (ket_01 + ket_10) / np.sqrt(2)
    print(f"|Ψ⁺⟩ = (|01⟩ + |10⟩)/√2 = {np.round(psi_plus.flatten(), 4)}")
    
    # |Ψ⁻⟩ = (|01⟩ - |10⟩)/√2
    psi_minus = (ket_01 - ket_10) / np.sqrt(2)
    print(f"|Ψ⁻⟩ = (|01⟩ - |10⟩)/√2 = {np.round(psi_minus.flatten(), 4)}")
    
    print("\n--- Bell State Orthonormality ---")
    bell_states = [phi_plus, phi_minus, psi_plus, psi_minus]
    names = ['|Φ⁺⟩', '|Φ⁻⟩', '|Ψ⁺⟩', '|Ψ⁻⟩']
    
    for i, (s1, n1) in enumerate(zip(bell_states, names)):
        for j, (s2, n2) in enumerate(zip(bell_states, names)):
            if j >= i:
                inner = np.dot(s1.conj().T, s2)[0, 0]
                if i == j:
                    print(f"⟨{n1}|{n2}⟩ = {inner.real:.0f} (normalized)")
                else:
                    print(f"⟨{n1}|{n2}⟩ = {inner.real:.0f} (orthogonal)")


def demo_entanglement_verification():
    """Verify entanglement of Bell states."""
    print("\n" + "=" * 60)
    print("ENTANGLEMENT VERIFICATION")
    print("=" * 60)
    
    def is_separable(state, tol=1e-10):
        """Check if state is separable using Schmidt decomposition."""
        mat = state.reshape(2, 2)
        _, s, _ = np.linalg.svd(mat)
        return np.sum(np.abs(s) > tol) == 1
    
    ket_00 = tensor(ket_0, ket_0)
    ket_01 = tensor(ket_0, ket_1)
    ket_10 = tensor(ket_1, ket_0)
    ket_11 = tensor(ket_1, ket_1)
    
    states = [
        ('|00⟩ (product)', ket_00),
        ('|++⟩ (product)', tensor((ket_0+ket_1)/np.sqrt(2), (ket_0+ket_1)/np.sqrt(2))),
        ('|Φ⁺⟩ (Bell)', (ket_00 + ket_11)/np.sqrt(2)),
        ('|Ψ⁻⟩ (Bell)', (ket_01 - ket_10)/np.sqrt(2)),
    ]
    
    print("\nSeparability test:")
    for name, state in states:
        sep = is_separable(state)
        status = "SEPARABLE (product)" if sep else "ENTANGLED"
        print(f"  {name}: {status}")


def plot_cnot_action():
    """Visualize CNOT action on basis states."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.patch.set_facecolor('#1a1a2e')
    fig.suptitle('CNOT Gate Action on Basis States', fontsize=16, fontweight='bold', color='white')
    
    basis_labels = ['|00⟩', '|01⟩', '|10⟩', '|11⟩']
    
    inputs = [
        tensor(ket_0, ket_0),
        tensor(ket_0, ket_1),
        tensor(ket_1, ket_0),
        tensor(ket_1, ket_1),
    ]
    
    for ax, (inp, label) in zip(axes.flat, zip(inputs, basis_labels)):
        ax.set_facecolor('#0a0a0a')
        
        output = np.dot(CNOT, inp)
        
        # Input probabilities
        in_probs = np.abs(inp.flatten())**2
        out_probs = np.abs(output.flatten())**2
        
        x = np.arange(4)
        width = 0.35
        
        bars1 = ax.bar(x - width/2, in_probs, width, label='Input', color='#4ecdc4', alpha=0.7)
        bars2 = ax.bar(x + width/2, out_probs, width, label='Output', color='#ff6b6b', alpha=0.7)
        
        ax.set_xticks(x)
        ax.set_xticklabels(basis_labels, color='white')
        ax.set_ylim(0, 1.2)
        ax.set_ylabel('Probability', color='white')
        ax.set_title(f'Input: {label}', fontsize=12, fontweight='bold', color='#64ffda')
        ax.tick_params(colors='white')
        ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('cnot_action.png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    print("\nSaved: cnot_action.png")
    plt.show()


def plot_bell_state_creation():
    """Visualize Bell state creation step by step."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('#1a1a2e')
    fig.suptitle('Bell State |Φ⁺⟩ Creation: |00⟩ → H⊗I → CNOT → |Φ⁺⟩', 
                fontsize=14, fontweight='bold', color='white')
    
    basis_labels = ['|00⟩', '|01⟩', '|10⟩', '|11⟩']
    
    # Initial state |00⟩
    ket_00 = tensor(ket_0, ket_0)
    
    # After H on first qubit
    H_I = tensor(H, I)
    after_H = np.dot(H_I, ket_00)
    
    # After CNOT
    bell_state = np.dot(CNOT, after_H)
    
    states = [
        ('Initial: |00⟩', ket_00),
        ('After H⊗I: (|00⟩+|10⟩)/√2', after_H),
        ('After CNOT: |Φ⁺⟩', bell_state),
    ]
    
    for ax, (title, state) in zip(axes, states):
        ax.set_facecolor('#0a0a0a')
        
        probs = np.abs(state.flatten())**2
        colors = ['#64ffda' if p > 0.01 else '#333' for p in probs]
        
        bars = ax.bar(basis_labels, probs, color=colors, edgecolor='white', linewidth=1)
        
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Probability', color='white')
        ax.set_title(title, fontsize=11, fontweight='bold', color='#64ffda')
        ax.tick_params(colors='white')
        
        for bar, prob in zip(bars, probs):
            if prob > 0.01:
                ax.text(bar.get_x() + bar.get_width()/2., prob + 0.02,
                       f'{prob:.2f}', ha='center', va='bottom', 
                       fontsize=10, color='white')
    
    plt.tight_layout()
    plt.savefig('bell_state_creation.png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    print("Saved: bell_state_creation.png")
    plt.show()


def plot_controlled_gate_matrices():
    """Visualize CNOT and CZ matrices."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('#1a1a2e')
    fig.suptitle('Controlled Gate Matrices', fontsize=16, fontweight='bold', color='white')
    
    gates = [('CNOT', CNOT), ('CZ', CZ)]
    
    for ax, (name, gate) in zip(axes, gates):
        ax.set_facecolor('#0a0a0a')
        
        # Plot matrix
        im = ax.imshow(np.real(gate), cmap='RdYlGn', vmin=-1, vmax=1)
        
        # Add text annotations
        for i in range(4):
            for j in range(4):
                val = gate[i, j]
                text = f'{int(val.real)}' if val.imag == 0 else f'{val}'
                color = 'white' if abs(val) > 0.5 else 'gray'
                ax.text(j, i, text, ha='center', va='center', 
                       fontsize=14, fontweight='bold', color=color)
        
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels(['|00⟩', '|01⟩', '|10⟩', '|11⟩'], color='white')
        ax.set_yticklabels(['|00⟩', '|01⟩', '|10⟩', '|11⟩'], color='white')
        ax.set_title(name, fontsize=14, fontweight='bold', color='#64ffda')
    
    plt.tight_layout()
    plt.savefig('controlled_matrices.png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    print("Saved: controlled_matrices.png")
    plt.show()


def main():
    """Main function."""
    print("\n" + "=" * 60)
    print("CONTROLLED GATES")
    print("CNOT, CZ, and Entanglement Creation")
    print("=" * 60)
    
    demo_cnot_basics()
    demo_cz_gate()
    demo_cnot_cz_relation()
    demo_bell_states()
    demo_entanglement_verification()
    
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    plot_cnot_action()
    plot_bell_state_creation()
    plot_controlled_gate_matrices()
    
    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("""
1. CNOT flips target if control is |1⟩
2. CZ adds phase -1 only to |11⟩
3. CZ is symmetric (control/target interchangeable)
4. CNOT = (I⊗H)·CZ·(I⊗H)
5. H + CNOT creates Bell states (entanglement)
6. Bell states are maximally entangled
7. CNOT is essential for universal quantum computing
8. Controlled gates enable quantum algorithms
    """)


if __name__ == "__main__":
    main()
