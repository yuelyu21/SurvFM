# Installation

## Install from GitHub

Clone the public repository and install the core package with a lightweight
local backbone:

```bash
git clone https://github.com/yuelyu21/SurvFM.git
cd SurvFM
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[sklearn]"
```

For a non-editable installation directly from GitHub:

```bash
python -m pip install "survfm-rmst[sklearn] @ git+https://github.com/yuelyu21/SurvFM.git"
```

Run the environment diagnostic without exposing credential values:

```bash
survfm-rmst diagnose
survfm-rmst backbones
```

## Optional installations

Install only the environment needed for a backbone:

```bash
python -m pip install -e ".[tabpfn]"
python -m pip install -e ".[tabicl]"
python -m pip install -e ".[tabdpt]"
python -m pip install -e ".[tabh2o]"
```

MITRA is large enough that a separate environment is recommended:

```bash
python -m venv .venv-mitra
source .venv-mitra/bin/activate
python -m pip install -e ".[mitra]"
```

!!! note "Model access"
    Some foundation-model checkpoints require license acceptance, login or a
    first-use download. Follow each model provider's official instructions and
    pin the package and checkpoint for a reproducible analysis.

## Documentation environment

```bash
python -m venv .venv-docs
source .venv-docs/bin/activate
python -m pip install -e ".[docs]"
mkdocs serve --dev-addr 127.0.0.1:8000
```

The command serves a local documentation preview.
