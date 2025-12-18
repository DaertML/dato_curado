import io
import sys
from contextlib import redirect_stdout, redirect_stderr

class PythonExecutorBridge:
    """
    Simulates the interface to a Python code execution environment.
    
    This executor runs the generated Python code and test case in a sandboxed 
    (though simple, for this simulation) environment and captures the output.
    """

    def __init__(self, timeout=5):
        self.timeout = timeout 
        print(f"Python Executor Bridge initialized with timeout: {timeout}s")

    def execute_code_and_test(self, solution_code: str, test_code: str) -> tuple[bool, str]:
        """
        Executes the combined code and checks for successful execution of the test.
        
        Args:
            solution_code: The function/class definition to be tested.
            test_code: The code to execute the solution (e.g., assert statements).
            
        Returns:
            A tuple: (True if execution was successful and test passed, 
                      Output/Error message string).
        """
        
        full_code = solution_code + "\n\n" + test_code

        # Use io.StringIO to capture stdout and stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        try:
            # Redirect stdout/stderr and execute
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # Execute the code in a local namespace
                exec(full_code, {})
                
            output = stdout_capture.getvalue()
            errors = stderr_capture.getvalue()
            
            if errors.strip():
                print(f"[EXECUTION SIMULATION] Code ran but produced errors/warnings: {errors.strip()[:100]}...")
                return False, f"RUNTIME ERROR/WARNING:\n{errors.strip()}"

            # Simple heuristic for success: If no exception was raised and no stderr output, assume success.
            # A proper test should use unittest or pytest and check for specific assertions.
            if not output.strip() and not errors.strip():
                 # No output means the assertions (which don't typically print on success) passed silently.
                print("\n[EXECUTION SIMULATION] Test ran successfully (no output or errors).")
                return True, "Code executed and tests passed successfully."
            else:
                print(f"\n[EXECUTION SIMULATION] Test ran successfully with output: {output.strip()[:100]}...")
                return True, f"Code executed successfully.\nOutput:\n{output.strip()}"

        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            print(f"\n[EXECUTION SIMULATION] EXECUTION FAILED: {error_msg}")
            return False, error_msg