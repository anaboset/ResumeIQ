import os
import tempfile
import atexit
from pathlib import Path

_TEMP_FILES: set[str] = set()

def cleanup_temp_files():
    """Clean up all temporary files on exit."""
    for path in _TEMP_FILES.copy():
        try:
            os.unlink(path)
            _TEMP_FILES.discard(path)
        except Exception:
            pass

atexit.register(cleanup_temp_files)

def save_uploaded_file(uploaded_file, max_size_mb: int = 10) -> str:
    """Save Streamlit UploadedFile to a temp file with proper cleanup."""
    # Validate file size
    if uploaded_file.size > max_size_mb * 1024 * 1024:
        raise ValueError(f"File exceeds {max_size_mb}MB limit")
    
    # Whitelist allowed extensions
    ALLOWED_EXTS = {'.pdf', '.txt', '.docx', '.md'}
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in ALLOWED_EXTS:
        raise ValueError(f"File type {suffix} not allowed")
    
    # Create temp file with proper cleanup
    with tempfile.NamedTemporaryFile(
        mode='wb',
        delete=False,
        suffix=suffix,
        dir=tempfile.gettempdir()
    ) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    
    _TEMP_FILES.add(tmp_path)
    return tmp_path

