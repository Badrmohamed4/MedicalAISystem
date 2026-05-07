import sys
import os

print("1. Starting Verification Script...")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("2. Importing ModelWrapper...")
try:
    from medical_chatbot.utils.image_processor import ModelWrapper
    print("   Import successful.")
except Exception as e:
    print(f"   Import FAILED: {e}")
    sys.exit(1)

print("3. Initializing Wrapper...")
wrapper = ModelWrapper()

print("4. Triggering load...")
wrapper._load_models()

print(f"   TF Ready: {wrapper.tf_ready}")
print(f"   Torch Ready: {wrapper.torch_ready}")

if not wrapper.tf_ready and wrapper.torch_ready:
    print("VERIFICATION SUCCESS: Hybrid Mode Active (TF Disabled, Torch Enabled).")
else:
    print("VERIFICATION FAILED: Unexpected State.")
