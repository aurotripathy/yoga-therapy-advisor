import openai

MODEL = "gpt-3.5-turbo"  # the chatGPT model
system_role_dict = {"role": "system", "content": "You are a helpful assistant."}
alternating_user_assistant_role = []
messages = [system_role_dict]

quit_words = ["--quit", "--bye", "--exit"]


def create_context_as_dict(response):
    """creates the dict format"""
    return {"role": "assistant", "content": response}


def create_prompt_as_dict(user_input):
    """Typically a question from the user"""
    return {"role": "user", "content": user_input}


def stateful_append_context_and_prompt(messages, context, prompt):
    messages.append(context)
    messages.append(prompt)
    return messages


def stateless_context_and_prompt(messages, context, prompt):
    messages = [system_role_dict, context, prompt]
    return messages


def get_response_from_chatGPT(messages):
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=messages,
        temperature=0,
    )
    print(f"Total tokens usage: {response['usage']['total_tokens']}")
    return response["choices"][0]["message"]["content"]

def dump_messages(messages):
     print(f"Messages into ChatGPT:")
     for message in messages:
         print(f'{message}')

# Inits
user_input = ""
response = ""
stateful = True

# use a while loop to continuously prompt the user for input until they enter a '--quit' or equivalent
while user_input not in quit_words:
    user_input = input(
        "Enter a question for the assistant (or type '--quit' to exit): "
    )

    if user_input not in quit_words:
        print(f'"You entered: ", {user_input}')

    prompt = create_prompt_as_dict(user_input)
    context = create_context_as_dict(response)

    if stateful:
        messages = stateful_append_context_and_prompt(messages, context, prompt)
    else:
        messages = stateless_context_and_prompt(messages, context, prompt)
    
    dump_messages(messages)

    response = get_response_from_chatGPT(messages)
    print(f'ChatGPT:{response}')
