# Install PyTorch with CUDA for Windows (RTX 40-series), then project deps.
# Run from repo root AFTER:  .\ .venv\Scripts\Activate.ps1

$ErrorActionPreference = "Stop"

python -m pip install -U pip wheel
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -e .
pip install huggingface_hub datasets

Write-Host ""
Write-Host "Next: log into Hugging Face, then prepare data:"
Write-Host "  1) Open https://huggingface.co/settings/tokens  (create Read token)"
Write-Host "  2) Accept terms: https://huggingface.co/datasets/lmsys/chatbot_arena_conversations"
Write-Host "  3) huggingface-cli login"
Write-Host "  4) python scripts/01a_check_arena_access.py"
Write-Host "  5) python scripts/01_prepare_data.py"
Write-Host ""
pytest -q
