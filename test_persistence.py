import torch
import numpy as np
import os
import sys

# Core Modules
from meta_cognition import MetaLearner

def main():
    print(">>> PERSISTENCE VERIFICATION <<<")

    # 1. Create Dummy Brain
    brain = MetaLearner(input_dim=7, hidden_dim=64)
    torch.save(brain.state_dict(), "test_brain.pth")
    print("[OK] Saved test_brain.pth")

    # 2. Load Brain
    new_brain = MetaLearner(input_dim=7, hidden_dim=64)
    try:
        new_brain.load_state_dict(torch.load("test_brain.pth"))
        print("[OK] Loaded test_brain.pth successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load brain: {e}")
        return

    # 3. Cleanup
    os.remove("test_brain.pth")
    print("[OK] Cleanup complete.")
    print("\n[SUCCESS] The Meta-Learner persistence mechanism is functional.")

if __name__ == "__main__":
    main()
