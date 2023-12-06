# Python version
Follow the [instruction](https://hackmd.io/c7W5omRvToCPP6SjBLtKrA) to install [alfworld](https://github.com/alfworld/alfworld) and stable-baselines3 in python3.10.
```
pip install -r requirements.txt
```

# Quickstart
- `config_file_path` : file path of *base_config.yaml*, e.g., *alfworld/configs/base_config.yaml*
- `alfworld_data_path` : the upper directory path of *json_2.1.1*.
```
python train.py [config_file_path] [alfworld_data_path]
```