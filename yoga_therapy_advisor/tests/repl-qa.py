import openai

MODEL = "gpt-3.5-turbo" # the chatGPT model
system_role_dict =  {"role": "system", "content": "You are a helpful assistant."}
alternating_user_assistant_role = []
messages = [system_role_dict]

quit_words = ['--quit', '--bye', '--exit']

def create_context(response):
    """creates the dict format"""
    return {"role": "assistant", "content": response}

def create_prompt(user_input):
    return {"role": "user", "content": user_input}


def append_context_and_prompt(messages, context, prompt):
    messages.append(context)
    messages.append(prompt)
    return messages

def get_response_from_GPT(messsages):
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=messages,
    temperature=0,
    )
    return response['choices'][0]['message']['content']


# Inits
user_input = ''
response = ''

#use a while loop to continuously prompt the user for input until they enter a '--quit' or equivalent
while user_input not in quit_words:
    user_input = input("Enter a question for the assistant (or type '--quit' to exit): ")
    
    if user_input not in quit_words:
        print("You entered: ", user_input)

    prompt = create_prompt(user_input)
    context = create_context(response)
    messages = append_context_and_prompt(messages, context, prompt)
    response = get_response_from_GPT(messages)
    print(response)
    