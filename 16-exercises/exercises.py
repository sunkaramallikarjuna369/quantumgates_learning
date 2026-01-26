"""
Exercises - Interactive Practice Problems
=========================================

This script provides practice problems and solutions
for quantum gates concepts.

Requirements: numpy
Install: pip install numpy
"""

import numpy as np

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

# Basis states
ket_0 = np.array([[1], [0]], dtype=complex)
ket_1 = np.array([[0], [1]], dtype=complex)


def tensor(*args):
    """Compute tensor product of multiple matrices."""
    result = args[0]
    for m in args[1:]:
        result = np.kron(result, m)
    return result


def exercise_1():
    """Exercise 1: Pauli-X action."""
    print("=" * 60)
    print("EXERCISE 1: Pauli-X Gate")
    print("=" * 60)
    print("\nQuestion: What does X|0⟩ equal?")
    print("\nSolution:")
    result = np.dot(X, ket_0)
    print(f"X|0⟩ = {result.flatten()}")
    print("Answer: |1⟩")
    print("\nThe Pauli-X gate flips |0⟩ to |1⟩ (quantum NOT gate).")


def exercise_2():
    """Exercise 2: Hadamard superposition."""
    print("\n" + "=" * 60)
    print("EXERCISE 2: Hadamard Gate")
    print("=" * 60)
    print("\nQuestion: What is H|0⟩?")
    print("\nSolution:")
    result = np.dot(H, ket_0)
    print(f"H|0⟩ = {np.round(result.flatten(), 4)}")
    print(f"     = (1/√2)|0⟩ + (1/√2)|1⟩")
    print(f"     = |+⟩")
    print("\nThe Hadamard gate creates an equal superposition.")


def exercise_3():
    """Exercise 3: H squared."""
    print("\n" + "=" * 60)
    print("EXERCISE 3: H² = I")
    print("=" * 60)
    print("\nQuestion: What is H·H?")
    print("\nSolution:")
    result = np.dot(H, H)
    print(f"H² = \n{np.round(result, 4)}")
    print(f"\nEquals I: {np.allclose(result, I)}")
    print("\nApplying Hadamard twice returns the original state.")


def exercise_4():
    """Exercise 4: Gate relationships."""
    print("\n" + "=" * 60)
    print("EXERCISE 4: S² = Z")
    print("=" * 60)
    print("\nQuestion: What is S·S?")
    print("\nSolution:")
    result = np.dot(S, S)
    print(f"S² = \n{np.round(result, 4)}")
    print(f"\nEquals Z: {np.allclose(result, Z)}")
    print("\nThe S gate is a π/2 phase rotation; two S gates = Z (π rotation).")


def exercise_5():
    """Exercise 5: X = HZH."""
    print("\n" + "=" * 60)
    print("EXERCISE 5: X = H·Z·H")
    print("=" * 60)
    print("\nQuestion: Verify that X = H·Z·H")
    print("\nSolution:")
    result = np.dot(H, np.dot(Z, H))
    print(f"H·Z·H = \n{np.round(result, 4)}")
    print(f"\nEquals X: {np.allclose(result, X)}")
    print("\nThis shows the basis change relationship between X and Z.")


def exercise_6():
    """Exercise 6: State evolution."""
    print("\n" + "=" * 60)
    print("EXERCISE 6: State Evolution")
    print("=" * 60)
    print("\nQuestion: What is Z·H|0⟩?")
    print("\nSolution:")
    step1 = np.dot(H, ket_0)
    print(f"Step 1: H|0⟩ = {np.round(step1.flatten(), 4)} = |+⟩")
    
    step2 = np.dot(Z, step1)
    print(f"Step 2: Z|+⟩ = {np.round(step2.flatten(), 4)} = |-⟩")
    print("\nZ flips the phase of |1⟩, converting |+⟩ to |-⟩.")


def exercise_7():
    """Exercise 7: Matrix multiplication."""
    print("\n" + "=" * 60)
    print("EXERCISE 7: X·Z Product")
    print("=" * 60)
    print("\nQuestion: Calculate X·Z")
    print("\nSolution:")
    result = np.dot(X, Z)
    print(f"X·Z = \n{np.round(result, 4)}")
    print(f"\nCompare to iY = \n{np.round(1j * Y, 4)}")
    print(f"\nX·Z = iY: {np.allclose(result, 1j * Y)}")


def exercise_8():
    """Exercise 8: Bell state creation."""
    print("\n" + "=" * 60)
    print("EXERCISE 8: Bell State Creation")
    print("=" * 60)
    print("\nQuestion: Create |Φ⁺⟩ from |00⟩")
    print("\nSolution:")
    
    state = tensor(ket_0, ket_0)
    print(f"Initial: |00⟩ = {state.flatten()}")
    
    H_I = tensor(H, I)
    state = np.dot(H_I, state)
    print(f"After H⊗I: {np.round(state.flatten(), 4)}")
    
    state = np.dot(CNOT, state)
    print(f"After CNOT: {np.round(state.flatten(), 4)}")
    print("\nThis is |Φ⁺⟩ = (|00⟩ + |11⟩)/√2")


def exercise_9():
    """Exercise 9: Entanglement check."""
    print("\n" + "=" * 60)
    print("EXERCISE 9: Entanglement Check")
    print("=" * 60)
    print("\nQuestion: Is (|00⟩ + |01⟩)/√2 entangled?")
    print("\nSolution:")
    
    state = (tensor(ket_0, ket_0) + tensor(ket_0, ket_1)) / np.sqrt(2)
    print(f"State: {np.round(state.flatten(), 4)}")
    
    # Check if separable
    print("\nCan we write it as |ψ₁⟩⊗|ψ₂⟩?")
    print("(|00⟩ + |01⟩)/√2 = |0⟩ ⊗ (|0⟩+|1⟩)/√2 = |0⟩ ⊗ |+⟩")
    print("\nAnswer: NOT entangled! It's a product state.")


def exercise_10():
    """Exercise 10: Unitarity check."""
    print("\n" + "=" * 60)
    print("EXERCISE 10: Unitarity Verification")
    print("=" * 60)
    print("\nQuestion: Verify that H is unitary (H†H = I)")
    print("\nSolution:")
    
    product = np.dot(H.conj().T, H)
    print(f"H†H = \n{np.round(product, 4)}")
    print(f"\nEquals I: {np.allclose(product, I)}")
    print("\nH is unitary, as required for all quantum gates.")


def run_all_exercises():
    """Run all exercises."""
    print("\n" + "=" * 60)
    print("QUANTUM GATES EXERCISES")
    print("Practice Problems with Solutions")
    print("=" * 60)
    
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
    exercise_6()
    exercise_7()
    exercise_8()
    exercise_9()
    exercise_10()
    
    print("\n" + "=" * 60)
    print("CONGRATULATIONS!")
    print("=" * 60)
    print("""
You have completed all the exercises!

Key concepts practiced:
1. Pauli gates (X, Y, Z)
2. Hadamard gate and superposition
3. Gate relationships (H² = I, S² = Z)
4. Basis changes (X = HZH)
5. State evolution
6. Matrix multiplication
7. Bell state creation
8. Entanglement detection
9. Unitarity verification

Keep practicing and exploring quantum computing!
    """)


if __name__ == "__main__":
    run_all_exercises()
