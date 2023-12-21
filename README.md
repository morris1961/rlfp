# Python env setup
Follow the instructions to install alfworld and required dependencies in python 3.10.
1. Install alfworld and download game files.
```bash
pip install alfworld[full]
export ALFWORLD_DATA=<storage_path>
alfworld-download
```
2. Install dependencies and download spacy model.
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

# LLM API setup
### LLaMA2
1. Run on another computer, because this model will consume 11GB GPU VRAM. 
1. Install [text-generation-webui](https://github.com/oobabooga/text-generation-webui) and start.
    
    1.  Clone or [download](https://github.com/oobabooga/text-generation-webui/archive/refs/heads/main.zip) the repository.

    2. Run the `start_linux.sh`, `start_windows.bat`, `start_macos.sh`, or `start_wsl.bat` script depending on your OS.
2. Once the installation ends, browse to http://localhost:7860/
3. Click the **Model** tab.
4. Under Download custom model or LoRA, enter `TheBloke/Llama-2-13B-chat-GPTQ`.
5. Click **Download**.
The model will start downloading. Once it's finished it will say "Done".
6. In the top left, click the refresh icon next to **Model**.
7. In the **Model** dropdown, choose the model you just downloaded: `Llama-2-13B-chat-GPTQ`
8. `Ctrl + c` in terminal to close the webui.
9. Add `--api --public-api --model TheBloke_Llama-2-13B-chat-GPTQ` to `text-generation-webui/CMD_FLAGS.txt` and restart text-generation-webui using start script.
10. Copy the API URL shows in terminal `https://xxx-xxx-xxx-xxx.trycloudflare.com`.
11. Change `code/utils/api.py` line **15** variable **API_URL** to the URL.


### Bard, Gemini pro free version
1. Set the environment variable `PALM_API_KEY` with your PALM API key.
    - Get PALM API key: [link](https://makersuite.google.com/app/apikey)

```bash
export PALM_API_KEY=<your key>
```
### Bard, Bard2, Gemini pro paid version
1. Create [google cloud account](https://cloud.google.com/free). When you first create an account, you get a $300 free credit towards your compute/storage costs.
2. [Select or create a Google Cloud project](https://console.cloud.google.com/cloud-resource-manager). 
3. [Enable the Vertex AI API](https://console.cloud.google.com/flows/enableapi?apiid=aiplatform.googleapis.com)
4. install the [Cloud SDK](https://cloud.google.com/sdk)
    - Ubuntu example
        ```bash=
        sudo apt-get update
        sudo apt-get install apt-transport-https ca-certificates gnupg curl sudo
        curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
        echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
        sudo apt-get update && sudo apt-get install google-cloud-cli
        gcloud init
        ```
5. Create local credentials.
    ```bash
    gcloud auth application-default login
    ```
# Quickstart
```bash
cd code
```
### Training
```bash
python train.py base_config.yaml
```

### Evaluate
```bash
python eval.py eval_config.yaml
```

### Run Baselines
```bash
bash baselines.sh
```
