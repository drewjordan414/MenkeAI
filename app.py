import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from dotenv import load_dotenv
import re

load_dotenv()
# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("satkinson/DialoGPT-small-marvin")
model = AutoModelForCausalLM.from_pretrained("satkinson/DialoGPT-small-marvin")

# Define the prompt
# prompt = ("You are an AI designed to emulate Menke, a person known for being a great friend, with some interesting always on his mind. When someone asks your name, you will say my name is Menke "
#           "You lack common sense, but thats what makes you great. You are always saying something that doesnt quite align with common sense. "
#           "You often say things that are unexpectedly out of context, "
#           "prompting others to wonder about the relevance of your comments. Your replies should be typically "
#           "short, mirroring Menke's unique manner of speaking. Additionally, infuse your interactions with "
#           "a demeanor that resembles a stoner's laid-back and seemingly disconnected attitude. Remember, your "
#           "goal is to be a quirky, sometimes puzzling approach to everyday "
#           "conversations, making your responses interestingly unusual and unexpectedly offbeat.")

prompt = os.environ.get('PROMPT')

# Initialize global variable for chat history
chat_history_ids = None

# def chatbot_response(user_input, temperature=0.7):
#     global chat_history_ids
    
#     # If starting a new conversation, prepend the prompt
#     if chat_history_ids is None:
#         chat_history_ids = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors='pt')

#     # Tokenize and encode user input
#     new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

#     # Concatenate new user input with chat history (including the prompt)
#     bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)

#     # Generate a response
#     chat_history_ids = model.generate(
#         bot_input_ids, 
#         max_length=1000, 
#         pad_token_id=tokenizer.eos_token_id, 
#         temperature=temperature
#     )

#     # Decode only the new tokens (the actual response), excluding the prompt and the previous conversation
#     response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    
#     # Update chat history with the new response
#     chat_history = tokenizer.decode(chat_history_ids[0], skip_special_tokens=True)
    
#     return chat_history, response
def chatbot_response(user_input, temperature=0.7):
    global chat_history_ids
    
    # Check if the user is asking for the bot's name
    if re.search(r"what'?s? your name", user_input, re.IGNORECASE):
        return "Dude, I'm Menke.", "Dude, I'm Menke."

    # If starting a new conversation, prepend the prompt
    if chat_history_ids is None:
        chat_history_ids = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors='pt')

    # Tokenize and encode user input
    new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    # Concatenate new user input with chat history (including the prompt)
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)

    # Generate a response
    chat_history_ids = model.generate(
        bot_input_ids, 
        max_length=1000, 
        pad_token_id=tokenizer.eos_token_id, 
        temperature=temperature
    )

    # Decode only the new tokens (the actual response), excluding the prompt and the previous conversation
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    
    # Update chat history with the new response
    chat_history = tokenizer.decode(chat_history_ids[0], skip_special_tokens=True)
    
    return chat_history, response

def clear_chat():
    global chat_history_ids
    chat_history_ids = None
    return "", ""

# Define the layout using Gradio's Blocks interface
with gr.Blocks() as demo:
    gr.Markdown("MenkeAI Chatbot")
    with gr.Row():
        with gr.Column(scale=1):
            chat_history_area = gr.Textbox(label="Chat History", lines=10, interactive=False)
        with gr.Column(scale=2):
            chat_dialogue_area = gr.Textbox(label="Chat Dialogue", lines=10, interactive=False)
    with gr.Row():
        user_input = gr.Textbox(label="User Chat Input")
        submit_button = gr.Button("Submit")
        clear_button = gr.Button("Clear")
        temperature_slider = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.7, label="Temperature")
    submit_button.click(chatbot_response, inputs=[user_input, temperature_slider], outputs=[chat_history_area, chat_dialogue_area])
    clear_button.click(clear_chat, inputs=[], outputs=[chat_history_area, chat_dialogue_area])

# Run the Gradio app
demo.launch()



# Notes
#This setup works well for single-user interactions. If multiple users interact with the model simultaneously, each will need a separate chat_history_ids to maintain different conversation states.
#The model's response might not strictly adhere to the prompt's character description in every instance, as it depends on the model's training and inherent capabilities.




