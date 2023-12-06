import os
from openai import OpenAI
import google.generativeai as palm

# OPENAI
# export OPENAI_API_KEY='["key", "key", "key", "key"]'
openai_api_keys = eval(os.environ['OPENAI_API_KEY'])
len_keys = len(openai_api_keys)
request_times = 0
# PALM
palm.configure(api_key=os.environ['PALM_API_KEY'])
models = [m for m in palm.list_models() if 'generateText' in m.supported_generation_methods]
palm_model = models[0].name
settings = {1: 4, 2: 4, 3: 4, 4: 4, 5: 4, 6: 4} # set all 6 categories to block none

def llm(prompt, stop=["\n"], model="gpt-3.5", max_tokens=100, temperature=0.0, top_p=1.0):
    if model == "gpt-3.5":
        global request_times
        request_times += 1
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        client = OpenAI(
            api_key=openai_api_keys[request_times % len_keys],
            base_url='https://api.chatanywhere.cn/v1',
        )
        response = client.chat.completions.create(
            model='gpt-3.5-turbo',
            messages=messages,
            max_tokens=max_tokens,
            stop=stop,
            temperature=temperature,
        )
        return response.choices[0].message.content
    elif model == "bard":
        completion = palm.generate_text(
            model=palm_model,
            prompt=prompt,
            temperature=temperature,
            max_output_tokens=max_tokens,
            stop_sequences=stop,
            safety_settings=settings,
            top_p=top_p,
        )
        return completion.result
    
def get_answer(prompt):
    for i in range(5):
        try:
            answers = [
                llm(prompt, model="bard"), 
                llm(prompt, model="bard"), 
                llm(prompt, model="bard")
            ]
            return answers
        except Exception as e:
            print(e)
            continue
    return []

if __name__ == "__main__":
    prompt = "Agent: go to the kitchen and pick up the apple. Then go to the bedroom and put it on the bed.\n\nHere is an example, you are Agent:\n\n"
    print(llm(prompt, model="gpt-3.5"))
    print(llm(prompt, model="bard"))
    print(get_answer(prompt))
