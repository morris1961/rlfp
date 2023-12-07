# Python version
1. Follow the [instruction](https://hackmd.io/c7W5omRvToCPP6SjBLtKrA) to install [alfworld](https://github.com/alfworld/alfworld) in python3.10

2. ```bash
    pip install -r requirements.txt
    ```
Set the environment variable `OPENAI_API_KEY` with your OpenAI API key and `PALM_API_KEY` with your PALM API key.
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

python train.py [config_file_path]
