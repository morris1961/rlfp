import os
import requests
from openai import OpenAI
import google.generativeai as palm
from vertexai.preview.language_models import TextGenerationModel
from vertexai.preview.generative_models import GenerativeModel

# PALM
palm.configure(api_key=os.environ['PALM_API_KEY'])
# set all 10 categories to block none
old_settings = {0: 4, 1: 4, 2: 4, 3: 4, 4: 4, 5: 4, 6: 4}
new_settings = {7: 4, 8: 4, 9: 4, 10: 4}
vertex_settings = {0: 4, 1: 4, 2: 4, 3: 4, 4: 4}
# text generation webui api url
API_URL = "https://smooth-looking-batteries-dual.trycloudflare.com"

def llm(prompt, stop=["\n"], model="bardfree", max_tokens=100, temperature=0.0, top_p=1.0):
    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]
    config = {
        'temperature': temperature,
        'max_output_tokens': max_tokens,
        'stop_sequences': stop,
        'top_p': top_p,
    }
    if model == "llama2":
        url = f"{API_URL}/v1/chat/completions"
        headers = {
            "Content-Type": "application/json"
        }
        data = {
            "mode": "instruct",
            "messages": messages,
            "max_tokens": max_tokens,
            "stop": stop,
        }
        response = requests.post(url, headers=headers, json=data, verify=False)
        assistant_message = response.json()['choices'][0]['message']['content']
        return assistant_message
    elif model == "bardfree":
        completion = palm.generate_text(
            model='models/text-bison-001',
            prompt=prompt,
            temperature=temperature,
            max_output_tokens=max_tokens,
            stop_sequences=stop,
            safety_settings=old_settings,
            top_p=top_p,
        )
        return completion.result
    elif model == "bard":
        model = TextGenerationModel.from_pretrained("text-bison@001")
        response = model.predict(
            prompt,
            temperature=temperature,
            max_output_tokens=max_tokens,
            stop_sequences=stop,
            top_p=top_p,
        )
        return response.text
    elif model == "bard2":
        model = TextGenerationModel.from_pretrained("text-bison@002")
        response = model.predict(
            prompt,
            temperature=temperature,
            max_output_tokens=max_tokens,
            stop_sequences=stop,
            top_p=top_p,
        )
        return response.text
    elif model == "gemini":
        model = palm.GenerativeModel('gemini-pro')
        response = model.generate_content(
            contents=prompt,
            generation_config=config,
            safety_settings=new_settings,
        )
        # model = GenerativeModel("gemini-pro")
        # response = model.generate_content(
        #     contents=prompt,
        #     generation_config=config,
        #     safety_settings=vertex_settings,
        # )
        return response.text
    
def get_answer(prompt, LLM_model_name):
    for i in range(5):
        try:
            answers = [llm(prompt, model=model_name) for model_name in LLM_model_name]
            return answers
        except Exception as e:
            print(e)
            continue
    return []

if __name__ == "__main__":
    prompt = "Agent: go to the kitchen and pick up the apple. Then go to the bedroom and put it on the bed.\n\nHere is an example, you are Agent:\n\n"
    print(llm(prompt, model="llama2"))
    print(llm(prompt, model="bard"))
    print(llm(prompt, model="bard2"))
    print(llm(prompt, model="gemini"))
    print(get_answer(prompt))