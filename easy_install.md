# Linux/MacOS one line install
```bash
git clone https://github.com/Vxtzq/CrowdGPT/ && cd CrowdGPT && \
REQ="requirements.txt"; \
if command -v nvidia-smi >/dev/null 2>&1; then REQ="requirements_cuda.txt"; \
elif command -v rocminfo >/dev/null 2>&1 || [ -d /opt/rocm ]; then REQ="requirements_rocm.txt"; \
fi; \
echo "🔧 Installing $REQ"; pip install -r "$REQ" && python crowdgpt.py
```

# Windows one line install (powershell)
```bash
git clone https://github.com/Vxtzq/CrowdGPT/; cd CrowdGPT; pip install -r requirements_directml.txt; python crowdgpt.py
```
