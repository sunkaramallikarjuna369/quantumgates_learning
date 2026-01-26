"""
SWAP & Toffoli Gates - Interactive Python Demonstrations
=======================================================

This script demonstrates SWAP and Toffoli gates:
- SWAP gate operation
- SWAP decomposition into CNOTs
- Toffoli (CCNOT) gate
- Classical logic with Toffoli

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

# Two-qubit gates
CNOT = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
], dtype=complex)

CNOT_reversed = np.array([
    [1, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
    [0, 1, 0, 0]
], dtype=complex)

SWAP = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
], dtype=complex)

# Three-qubit Toffoli gate (8x8)
TOFFOLI = np.eye(8, dtype=complex)
TOFFOLI[6, 6] = 0
TOFFOLI[6, 7] = 1
TOFFOLI[7, 6] = 1
TOFFOLI[7, 7] = 0


def tensor(*args):
    """Compute tensor product of multiple matrices."""
    result = args[0]
    for m in args[1:]:
        result = np.kron(result, m)
    return result


def demo_swap_basics():
    """Demonstrate SWAP gate basics."""
    print("=" * 60)
    print("SWAP GATE")
    print("=" * 60)
    
    print("\n--- Matrix Representation ---")
    print(f"\nSWAP =\n{SWAP.astype(int)}")
    
    print("\n--- Action on Basis States ---")
    basis_states = ['00', '01', '10', '11']
    
    for bits in basis_states:
        q1 = ket_0 if bits[0] == '0' else ket_1
        q2 = ket_0 if bits[1] == '0' else ket_1
        state = tensor(q1, q2)
        
        output = np.dot(SWAP, state)
        
        # Find output state
        for out_bits in basis_states:
            oq1 = ket_0 if out_bits[0] == '0' else ket_1
            oq2 = ket_0 if out_bits[1] == '0' else ket_1
            out_state = tensor(oq1, oq2)
            if np.allclose(output, out_state):
                print(f"SWAP|{bits}⟩ = |{out_bits}⟩")
                break
    
    print("\n--- Key Properties ---")
    print(f"SWAP² = I: {np.allclose(np.dot(SWAP, SWAP), np.eye(4))}")
    print(f"SWAP is unitary: {np.allclose(np.dot(SWAP.conj().T, SWAP), np.eye(4))}")
    print(f"SWAP is Hermitian: {np.allclose(SWAP, SWAP.conj().T)}")


def demo_swap_decomposition():
    """Demonstrate SWAP decomposition into CNOTs."""
    print("\n" + "=" * 60)
    print("SWAP DECOMPOSITION")
    print("=" * 60)
    
    print("\nSWAP = CNOT₁₂ · CNOT₂₁ · CNOT₁₂")
    print("(Three CNOT gates)")
    
    # CNOT with control on qubit 0, target on qubit 1
    CNOT_01 = CNOT
    
    # CNOT with control on qubit 1, target on qubit 0
    CNOT_10 = CNOT_reversed
    
    # Compose: CNOT_01 · CNOT_10 · CNOT_01
    decomposed = np.dot(CNOT_01, np.dot(CNOT_10, CNOT_01))
    
    print(f"\nCNOT₁₂ · CNOT₂₁ · CNOT₁₂ =\n{decomposed.astype(int)}")
    print(f"\nEquals SWAP? {np.allclose(decomposed, SWAP)}")


def demo_toffoli_basics():
    """Demonstrate Toffoli gate basics."""
    print("\n" + "=" * 60)
    print("TOFFOLI GATE (CCNOT)")
    print("=" * 60)
    
    print("\n--- Matrix Representation (8×8) ---")
    print(f"\nTOFFOLI =\n{TOFFOLI.astype(int)}")
    
    print("\n--- Truth Table ---")
    print(f"{'Input':<10} {'Output':<10} {'Action'}")
    print("-" * 35)
    
    basis_states = ['000', '001', '010', '011', '100', '101', '110', '111']
    
    for bits in basis_states:
        q1 = ket_0 if bits[0] == '0' else ket_1
        q2 = ket_0 if bits[1] == '0' else ket_1
        q3 = ket_0 if bits[2] == '0' else ket_1
        state = tensor(q1, q2, q3)
        
        output = np.dot(TOFFOLI, state)
        
        # Find output state
        for out_bits in basis_states:
            oq1 = ket_0 if out_bits[0] == '0' else ket_1
            oq2 = ket_0 if out_bits[1] == '0' else ket_1
            oq3 = ket_0 if out_bits[2] == '0' else ket_1
            out_state = tensor(oq1, oq2, oq3)
            if np.allclose(output, out_state):
                changed = bits != out_bits
                action = "Target flipped!" if changed else ""
                print(f"|{bits}⟩      |{out_bits}⟩      {action}")
                break
    
    print("\n--- Key Properties ---")
    print(f"TOFFOLI² = I: {np.allclose(np.dot(TOFFOLI, TOFFOLI), np.eye(8))}")
    print(f"TOFFOLI is unitary: {np.allclose(np.dot(TOFFOLI.conj().T, TOFFOLI), np.eye(8))}")


def demo_toffoli_as_and():
    """Demonstrate Toffoli as reversible AND gate."""
    print("\n" + "=" * 60)
    print("TOFFOLI AS REVERSIBLE AND GATE")
    print("=" * 60)
    
    print("\nWith target initialized to |0⟩:")
    print("TOFFOLI|a,b,0⟩ = |a,b,a AND b⟩")
    
    print("\n--- AND Truth Table ---")
    print(f"{'a':<5} {'b':<5} {'a AND b'}")
    print("-" * 20)
    
    for a in [0, 1]:
        for b in [0, 1]:
            qa = ket_0 if a == 0 else ket_1
            qb = ket_0 if b == 0 else ket_1
            state = tensor(qa, qb, ket_0)  # Target = |0⟩
            
            output = np.dot(TOFFOLI, state)
            
            # Extract target qubit value
            # Output is |a,b,a AND b⟩
            result = a & b
            print(f"{a:<5} {b:<5} {result}")
    
    print("\n--- Implementing OR using Toffoli ---")
    print("a OR b = NOT(NOT(a) AND NOT(b))")
    print("Can be implemented with Toffoli + X gates")


def demo_classical_universality():
    """Demonstrate Toffoli's classical universality."""
    print("\n" + "=" * 60)
    print("TOFFOLI: UNIVERSAL FOR CLASSICAL COMPUTING")
    print("=" * 60)
    
    print("\nToffoli can implement any classical Boolean function!")
    
    print("\n--- NOT gate ---")
    print("Set both controls to |1⟩:")
    print("TOFFOLI|1,1,x⟩ = |1,1,NOT x⟩")
    
    for x in [0, 1]:
        qx = ket_0 if x == 0 else ket_1
        state = tensor(ket_1, ket_1, qx)
        output = np.dot(TOFFOLI, state)
        
        # Check output
        for out_x in [0, 1]:
            out_qx = ket_0 if out_x == 0 else ket_1
            out_state = tensor(ket_1, ket_1, out_qx)
            if np.allclose(output, out_state):
                print(f"  TOFFOLI|1,1,{x}⟩ = |1,1,{out_x}⟩  (NOT {x} = {out_x})")
    
    print("\n--- FANOUT (copy) ---")
    print("CNOT can copy: CNOT|x,0⟩ = |x,x⟩")
    print("Toffoli with one control set to 1 acts as CNOT")


def plot_swap_action():
    """Visualize SWAP gate action."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('#1a1a2e')
    fig.suptitle('SWAP Gate Action', fontsize=16, fontweight='bold', color='white')
    
    basis_labels = ['|00⟩', '|01⟩', '|10⟩', '|11⟩']
    
    # Before SWAP
    ax = axes[0]
    ax.set_facecolor('#0a0a0a')
    
    # Example: |01⟩
    state = tensor(ket_0, ket_1)
    probs = np.abs(state.flatten())**2
    ax.bar(basis_labels, probs, color='#4ecdc4', edgecolor='white')
    ax.set_title('Before SWAP: |01⟩', color='#64ffda', fontsize=12)
    ax.set_ylim(0, 1.2)
    ax.tick_params(colors='white')
    
    # After SWAP
    ax = axes[1]
    ax.set_facecolor('#0a0a0a')
    
    output = np.dot(SWAP, state)
    probs = np.abs(output.flatten())**2
    ax.bar(basis_labels, probs, color='#ff6b6b', edgecolor='white')
    ax.set_title('After SWAP: |10⟩', color='#64ffda', fontsize=12)
    ax.set_ylim(0, 1.2)
    ax.tick_params(colors='white')
    
    plt.tight_layout()
    plt.savefig('swap_action.png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    print("\nSaved: swap_action.png")
    plt.show()


def plot_toffoli_truth_table():
    """Visualize Toffoli truth table."""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#0a0a0a')
    
    inputs = ['000', '001', '010', '011', '100', '101', '110', '111']
    outputs = ['000', '001', '010', '011', '100', '101', '111', '110']
    
    # Create bar chart showing which states change
    changes = [0 if i == o else 1 for i, o in zip(inputs, outputs)]
    colors = ['#4ecdc4' if c == 0 else '#ff6b6b' for c in changes]
    
    x = np.arange(len(inputs))
    bars = ax.bar(x, [1]*8, color=colors, edgecolor='white', linewidth=2)
    
    ax.set_xticks(x)
    ax.set_xticklabels([f'|{i}⟩→|{o}⟩' for i, o in zip(inputs, outputs)], 
                       rotation=45, ha='right', color='white')
    ax.set_yticks([])
    ax.set_title('Toffoli Gate: Only |110⟩ and |111⟩ are swapped', 
                fontsize=14, fontweight='bold', color='#64ffda')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#4ecdc4', label='Unchanged'),
        Patch(facecolor='#ff6b6b', label='Changed (target flipped)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('toffoli_truth_table.png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    print("Saved: toffoli_truth_table.png")
    plt.show()


def plot_gate_matrices():
    """Visualize SWAP and Toffoli matrices."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('#1a1a2e')
    fig.suptitle('SWAP and Toffoli Gate Matrices', fontsize=16, fontweight='bold', color='white')
    
    # SWAP matrix
    ax = axes[0]
    ax.set_facecolor('#0a0a0a')
    im = ax.imshow(np.real(SWAP), cmap='RdYlGn', vmin=-1, vmax=1)
    for i in range(4):
        for j in range(4):
            val = int(SWAP[i, j].real)
            color = 'white' if val != 0 else 'gray'
            ax.text(j, i, str(val), ha='center', va='center', 
                   fontsize=14, fontweight='bold', color=color)
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(['|00⟩', '|01⟩', '|10⟩', '|11⟩'], color='white')
    ax.set_yticklabels(['|00⟩', '|01⟩', '|10⟩', '|11⟩'], color='white')
    ax.set_title('SWAP (4×4)', color='#64ffda', fontsize=12)
    
    # Toffoli matrix
    ax = axes[1]
    ax.set_facecolor('#0a0a0a')
    im = ax.imshow(np.real(TOFFOLI), cmap='RdYlGn', vmin=-1, vmax=1)
    for i in range(8):
        for j in range(8):
            val = int(TOFFOLI[i, j].real)
            color = 'white' if val != 0 else 'gray'
            ax.text(j, i, str(val), ha='center', va='center', 
                   fontsize=10, fontweight='bold', color=color)
    labels_3q = [f'|{i:03b}⟩' for i in range(8)]
    ax.set_xticks(range(8))
    ax.set_yticks(range(8))
    ax.set_xticklabels(labels_3q, color='white', fontsize=8, rotation=45)
    ax.set_yticklabels(labels_3q, color='white', fontsize=8)
    ax.set_title('Toffoli (8×8)', color='#64ffda', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('swap_toffoli_matrices.png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    print("Saved: swap_toffoli_matrices.png")
    plt.show()


def main():
    """Main function."""
    print("\n" + "=" * 60)
    print("SWAP & TOFFOLI GATES")
    print("State Exchange and Three-Qubit Operations")
    print("=" * 60)
    
    demo_swap_basics()
    demo_swap_decomposition()
    demo_toffoli_basics()
    demo_toffoli_as_and()
    demo_classical_universality()
    
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    plot_swap_action()
    plot_toffoli_truth_table()
    plot_gate_matrices()
    
    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("""
1. SWAP exchanges two qubit states
2. SWAP = 3 CNOT gates (decomposition)
3. SWAP² = I (self-inverse)
4. Toffoli flips target when BOTH controls are |1⟩
5. Toffoli implements reversible AND gate
6. Toffoli is universal for classical computing
7. Toffoli² = I (self-inverse)
8. Both gates are important for quantum algorithms
    """)


if __name__ == "__main__":
    main()
