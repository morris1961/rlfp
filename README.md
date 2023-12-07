# Python version
Follow the [instruction](https://hackmd.io/c7W5omRvToCPP6SjBLtKrA) to install [alfworld](https://github.com/alfworld/alfworld) and stable-baselines3 in python3.10.
```bash
pip install -r requirements.txt
```
Set the environment variable `OPENAI_API_KEY` with your OpenAI API key and `PALM_API_KEY` with your PALM API key.
```bash
export OPENAI_API_KEY="['key', 'key', 'key', ...]"
export PALM_API_KEY='key'
```

# Quickstart
- `config_file_path` : file path of *base_config.yaml*, e.g., *alfworld/configs/base_config.yaml*
- `alfworld_data_path` : the upper directory path of *json_2.1.1*.
```bash
python train.py [config_file_path] [alfworld_data_path]
```