"""
Run with:

```bash
flyte run --follow python-script hello.py --output-dir output
```
"""

import os


def main():
    print("Hello, world!")
    os.makedirs("output", exist_ok=True)
    with open("output/hello.txt", "w") as f:
        f.write("Hello, file!")


if __name__ == "__main__":
    main()
