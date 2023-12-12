# Python version
Follow the [instruction](https://hackmd.io/c7W5omRvToCPP6SjBLtKrA) to install [alfworld](https://github.com/alfworld/alfworld) in python3.10

# Dependencies
1. install dependencies and spacy
```bash
pip install -r requirements.txt
# install spacy dependency
python -m spacy download en_core_web_sm
```
2. Set the environment variable `OPENAI_API_KEY` with your OpenAI API key and `PALM_API_KEY` with your PALM API key.
    - Get PALM API key: [link](https://makersuite.google.com/app/apikey)

```bash
export OPENAI_API_KEY="['key', 'key', 'key', ...]"
export PALM_API_KEY='key'
```

# Datapath
```bash
export ALFWORLD_DATA=<storage_path>
alfworld-download # do only the first time to download dataset to <storage_path>
```

# Quickstart
- `config_file_path` : file path of *base_config.yaml*, e.g., *base_config.yaml*

```bash
python train.py [config_file_path]
```

# Run Baselines
```bash
bash baselines.sh
```
