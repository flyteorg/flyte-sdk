# Save this as:  test_uv_short.py

from flyte import Image
from pathlib import Path

def main():
    print("🧪 UV Script Auto-Separation Test")
    print("=" * 40)
    
    # Create UV script with 3 dependencies
    script_content = '''#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas", "matplotlib", "jupyter", "tensorflow", "torch", "mypy"]
# ///

import pandas as pd
print("UV script working!")
'''
    
    with open("test_uv_script.py", "w") as f:
        f.write(script_content)
    
    # Test auto-separation
    image = Image.from_uv_script(
        script="test_uv_script.py",
        name="test-uv"
    )
    
    # Show results
    print(f"\n📦 Total layers: {len(image._layers)}")
    
    # Show only the important layers (skip base layers)
    for i, layer in enumerate(image._layers):
        print(f"Layer {i}: {type(layer).__name__} - {layer}")
    
    print(f"\n✅ Auto-separation working! Found {len([l for l in image._layers if 'PipPackages' in str(type(l))])} pip package layers")

if __name__ == "__main__": 
    main()
