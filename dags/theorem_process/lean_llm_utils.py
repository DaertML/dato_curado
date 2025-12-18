import re
import os
from ollama import chat # Real ollama chat import
from lean_prover_bridge import LeanProverBridge

# --- LLM Configuration ---
# NOTE: Ensure the Ollama server is running and this model is pulled locally.
OLLAMA_MODEL = "qwen3:8b" 

prompt_lean_generation = """
You are a formal mathematics assistant. Your task is to translate a natural language mathematical exercise into Lean 4 code.
Your output MUST contain a reasoning section, the Lean code for the theorem, and its proof, all enclosed in the specified delimiters.

CRITICAL INSTRUCTION: All generated code MUST start with 'import Mathlib.Data.Nat.Basic' if it involves natural numbers (Nat) or arithmetic.

Use the following exact delimiters:
1. The **Reasoning Section**: A natural language explanation of the proof strategy (how you would prove it informally).
   - Start of Reasoning: [REASONING_NL]
   - End of Reasoning: [END_REASONING_NL]
2. The **Theorem Definition**: The Lean statement of the claim, including the final ':= by'.
   - Start of Theorem: [THEOREM_DEF]
   - End of Theorem: [END_THEOREM_DEF]
3. The **Proof Code**: The Lean tactics that prove the theorem.
   - Start of Proof: [PROOF_CODE]
   - End of Proof: [END_PROOF_CODE]

Example:
Question: Prove that for all natural numbers \(n\), \(n + 0 = n\).
Output:
[REASONING_NL]
The proof is straightforward. We will use the built-in Lean theorem 'Nat.add_zero' which states that adding zero to any natural number leaves it unchanged. After applying this theorem, the goal will be directly provable by reflexivity ('rfl').
[END_REASONING_NL]

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

def extract_lean_code_and_reasoning(response: str) -> tuple[str, str, str, str]:
    """
    Extracts the theorem definition, proof code, reasoning, and the combined
    lean code from the LLM response using delimiters.
    """
    
    # Use re.DOTALL and allow for potential surrounding whitespace
    reasoning_match = re.search(r"\[REASONING_NL\]\s*(.*?)\s*\[END_REASONING_NL\]", response, re.DOTALL)
    theorem_match = re.search(r"\[THEOREM_DEF\]\s*(.*?)\s*\[END_THEOREM_DEF\]", response, re.DOTALL)
    proof_match = re.search(r"\[PROOF_CODE\]\s*(.*?)\s*\[END_PROOF_CODE\]", response, re.DOTALL)
    
    # Extract the content, stripping outer whitespace
    reasoning_nl = reasoning_match.group(1).strip() if reasoning_match else ""
    theorem_def = theorem_match.group(1).strip() if theorem_match else ""
    proof_code = proof_match.group(1).strip() if proof_match else ""

    # IMPROVEMENT: Combine theorem and proof for lean code extraction
    # This creates the complete, runnable Lean file content.
    if theorem_def and proof_code:
        # The theorem definition includes ':= by', so we just append the proof code and 'done'
        lean_file_code = f"{theorem_def}\n  {proof_code}\n"
        # Often a proof ends with 'done' in Lean 4 to signal the end of a tactic block, 
        # but the LLM might omit it. We'll leave it as is for maximum flexibility, 
        # assuming 'rfl' or another final tactic suffices.
    else:
        lean_file_code = ""
    
    return reasoning_nl, theorem_def, proof_code, lean_file_code


def check_lean_solution(prover: LeanProverBridge, question: str) -> dict:
    """
    1. Calls the LLM to generate the Lean theorem, proof, and reasoning.
    2. Calls the Lean Prover (simulated) to verify the proof against the theorem.
    3. Returns the result including the reasoning and full Lean code.
    """
    
    # 1. Generate Lean code and reasoning from natural language
    llm_response = run_ollama(OLLAMA_MODEL, prompt_lean_generation, question['question'])
    reasoning_nl, theorem_def, proof_code, lean_file_code = extract_lean_code_and_reasoning(llm_response)
    
    print("-" * 50)
    print(f"NL Question: {question}")
    
    if not theorem_def or not proof_code or not reasoning_nl:
        print("!! ERROR: Failed to extract complete Lean components (Theorem, Proof, or Reasoning) from LLM.")
        return {
            "question": question,
            "reasoning": reasoning_nl,
            "theorem": theorem_def,
            "proof": proof_code,
            "lean_code": lean_file_code, # New field
            "valid": False,
            "error": "Parsing failed or missing component",
            "llm_output": llm_response # Include full LLM output for debugging
        }
        
    # 2. Check validity using the Prover Bridge (simulated verification)
    # The prover typically checks the combined code for verification
    is_valid = prover.check_full_theorem_proof(theorem_def, proof_code)
    
    # 3. Return the comprehensive result
    return {
        "question": question,
        "reasoning": reasoning_nl, # New field
        "theorem": theorem_def,
        "proof": proof_code,
        "lean_code": lean_file_code, # New field
        "valid": is_valid
    }

# NOTE: The prover.check_full_theorem_proof function in a real scenario would
# likely be modified to accept the full `lean_file_code` for verification, 
# though the current structure is retained for compatibility with the original bridge.