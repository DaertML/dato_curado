import re
import os
from ollama import chat # Real ollama chat import
from lean_prover_bridge import LeanProverBridge

# --- LLM Configuration ---
# NOTE: Ensure the Ollama server is running and this model is pulled locally.
OLLAMA_MODEL = "qwen3:8b" 

prompt_lean_generation = """
You are a formal mathematics assistant. Your task is to translate a natural language mathematical exercise into Lean 4 code.
Your output MUST contain the Lean code for the theorem and its proof, enclosed in the specified delimiters.

CRITICAL INSTRUCTION: All generated code MUST start with 'import Mathlib.Data.Nat.Basic' if it involves natural numbers (Nat) or arithmetic.

Use the following exact delimiters:
1. The **Theorem Definition**: The Lean statement of the claim, including the final ':= by'.
   - Start of Theorem: [THEOREM_DEF]
   - End of Theorem: [END_THEOREM_DEF]
2. The **Proof Code**: The Lean tactics that prove the theorem.
   - Start of Proof: [PROOF_CODE]
   - End of Proof: [END_PROOF_CODE]

Example:
Question: Prove that for all natural numbers \(n\), \(n + 0 = n\).
Output:
[THEOREM_DEF]
import Mathlib.Data.Nat.Basic
theorem add_zero_test (n : Nat) : n + 0 = n := by
[END_THEOREM_DEF]

[PROOF_CODE]
rw [Nat.add_zero]
rfl
[END_PROOF_CODE]
"""

# --- Core Utilities ---

def run_ollama(model, prompt, question):
    """Makes a real call to the Ollama server."""
    print(f"\n--- OLLAMA CALL ({model}) ---")
    response = chat(model=model, messages=[
        {
            'role': 'system',
            'content': prompt,
        },
        {
            'role': 'user',
            'content': question
        }
    ])
    # The actual response structure is complex, but the content is here:
    return response['message']['content']

def extract_lean_code(response: str) -> tuple[str, str]:
    """Extracts the theorem and proof code from the LLM response using delimiters."""
    
    # Use re.DOTALL and allow for potential surrounding whitespace
    theorem_match = re.search(r"\[THEOREM_DEF\]\s*(.*?)\s*\[END_THEOREM_DEF\]", response, re.DOTALL)
    proof_match = re.search(r"\[PROOF_CODE\]\s*(.*?)\s*\[END_PROOF_CODE\]", response, re.DOTALL)
    
    # Extract the content, stripping outer whitespace
    theorem_def = theorem_match.group(1).strip() if theorem_match else ""
    proof_code = proof_match.group(1).strip() if proof_match else ""
    
    return theorem_def, proof_code

def check_lean_solution(prover: LeanProverBridge, question: str) -> dict:
    """
    1. Calls the LLM to generate the Lean theorem and proof.
    2. Calls the Lean Prover (simulated) to verify the proof against the theorem.
    3. Returns the result.
    """
    
    # 1. Generate Lean code from natural language
    llm_response = run_ollama(OLLAMA_MODEL, prompt_lean_generation, question)
    theorem_def, proof_code = extract_lean_code(llm_response)
    
    print("-" * 50)
    print(f"NL Question: {question}")
    
    if not theorem_def or not proof_code:
        print("!! ERROR: Failed to extract complete Lean code from LLM.")
        return {
            "question": question,
            "theorem": theorem_def,
            "proof": proof_code,
            "valid": False,
            "error": "Parsing failed",
            "llm_output": llm_response # Include full LLM output for debugging
        }
        
    # 2. Check validity using the Prover Bridge (simulated verification)
    is_valid = prover.check_full_theorem_proof(theorem_def, proof_code)
    
    # 3. Return the comprehensive result
    return {
        "question": question,
        "theorem": theorem_def,
        "proof": proof_code,
        "valid": is_valid
    }