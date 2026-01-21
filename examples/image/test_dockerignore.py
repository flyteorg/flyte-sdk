import shutil
import tempfile
from pathlib import Path

from flyte import Image
from flyte._internal.imagebuild.utils import get_and_list_dockerignore

# Your dockerignore file
dockerignore_path = Path(__file__).parent / ".dockerignore"

# Test directory with files that should be ignored
test_dir = Path(tempfile.mkdtemp())

(test_dir / ".cache").mkdir()
(test_dir / ".cache" / "test.txt").write_text("should be ignored")
(test_dir / "keep_me.py").write_text("should be kept")

(test_dir / "__pycache__").mkdir()
(test_dir / "__pycache__" / "test.pyc").write_text("should be ignored")

print(f"Test directory:  {test_dir}")
print(f"Files created: {list(test_dir.rglob('*'))}")

# Create an Image with dockerignore
image = Image.from_debian_base(install_flyte=False).with_dockerignore(dockerignore_path)

# Get ignore patterns from the Image (already transformed in utils.py!)
fnmatch_patterns = get_and_list_dockerignore(image)
print(f"\n🔍 Fnmatch patterns (ready to use): {fnmatch_patterns}")

# Copy with ignore patterns
dest_dir = Path(tempfile.mkdtemp())
shutil.copytree(
    test_dir,
    dest_dir / "copied",
    ignore=shutil.ignore_patterns(*fnmatch_patterns),
)

print(f"\n✅ Files copied to {dest_dir}/copied:")
for f in (dest_dir / "copied").rglob("*"):
    print(f"  - {f.relative_to(dest_dir / 'copied')}")

# Check what was skipped
print("\n❌ Files that should have been skipped:")

if (dest_dir / "copied" / ".cache").exists():
    print("  ⚠️  .cache/ was copied (BAD!)")
else:
    print("  ✅ .cache/ was skipped (GOOD!)")

if (dest_dir / "copied" / "__pycache__").exists():
    print("  ⚠️  __pycache__/ was copied (BAD!)")
else:
    print("  ✅ __pycache__/ was skipped (GOOD!)")

# Cleanup
shutil.rmtree(test_dir)
shutil.rmtree(dest_dir)
