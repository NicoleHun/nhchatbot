# suppress warnings
import warnings
warnings.filterwarnings("ignore")

# import libraries
import requests  # For making HTTP requests (e.g., downloading images)
import os        # For interacting with the operating system (e.g., file paths)
import argparse  # For parsing command-line arguments
from PIL import Image  # For image processing (opening, resizing, saving images)
import signal  # Add this import at the top with your other imports
import sys
import gradio as gr  # For creating web-based UIs easily
from together import Together  # For interacting with the Together API (LLM/image generation)
import textwrap  # For formatting/wrapping text output


## FUNCTION 1: This Allows Us to Prompt the AI MODEL
# -------------------------------------------------
def prompt_llm(prompt, with_linebreak=False):
    # This function allows us to prompt an LLM via the Together API

    # model
    model = "meta-llama/Meta-Llama-3-8B-Instruct-Lite"

    # Calculate the number of tokens
    tokens = len(prompt.split())

    # Make the API call
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    output = response.choices[0].message.content

    if with_linebreak:
        # Wrap the output
        wrapped_output = textwrap.fill(output, width=50)

        return wrapped_output
    else:
        return output


## FUNCTION 2: This Allows Us to Generate Images
# -------------------------------------------------
def gen_image(prompt, width=256, height=256):
    # This function allows us to generate images from a prompt
    response = client.images.generate(
        prompt=prompt,
        model="black-forest-labs/FLUX.1-schnell-Free",  # Using a supported model
        steps=2,
        n=1,
    )
    image_url = response.data[0].url
    image_filename = "image.png"

    # Download the image using requests instead of wget
    response = requests.get(image_url)
    with open(image_filename, "wb") as f:
        f.write(response.content)
    img = Image.open(image_filename)
    img = img.resize((height, width))

    return img


## Function 3: This Allows Us to Create a Chatbot
# -------------------------------------------------

def add_user_message(user_message, chat_history):
    # This function immediately adds the user message to the chat
    return "", chat_history + [(user_message, None)]

def format_chat_history(history):
    print("\n=== DEBUG: format_chat_history ===")
    print(f"History length: {len(history)}")
    print("Full history:", history)
    
    if not history:
        return "No previous conversation."
    
    formatted = []
    for user_msg, bot_msg in history[-3:]:  # Focus on last 3 exchanges
        if user_msg and bot_msg:
            formatted.append(f"User: {user_msg}\nRoasty: {bot_msg}")
    
    print("Formatted history:", formatted)
    print("=== END DEBUG ===\n")
    return "\n\n".join(formatted)

def clean_response(response):
    """Clean and format the emoji-structured response"""
    # Split the response into lines and clean up
    lines = [line.strip() for line in response.split('\n') if line.strip()]
    
    # Make sure each section is complete
    complete_lines = []
    current_line = ""
    
    for line in lines:
        # If this is a new section (starts with emoji), save the previous line
        if any(emoji in line for emoji in ['🔄', '💡', '💭', '➡️', '❓']):
            if current_line:
                complete_lines.append(current_line)
            current_line = line
        else:
            # Append to current line to handle multi-line sections
            current_line += " " + line.strip()
    
    # Don't forget to add the last line
    if current_line:
        complete_lines.append(current_line)
    
    return '\n'.join(complete_lines)

def bot_response_function(user_message, chat_history):
    # First, add the user message to chat history
    chat_history = chat_history + [(user_message, None)]
    
    # Limit chat history to last 3 exchanges for faster context processing
    recent_history = chat_history[-4:-1] if len(chat_history) > 4 else chat_history[:-1]
    formatted_history = format_chat_history(recent_history)

    chatbot_prompt = f"""
    You are Roasty, a brutally honest but emotionally intelligent chatbot who maintains engaging conversations.
    
    Recent conversation history:
    {formatted_history}

    Current message: "{user_message}"

    IMPORTANT: Structure your response using exactly these sections and emojis:

    🔄 [Brief acknowledgment of previous conversation or current situation with burtal humor roasting the user]
    💡 [Share a relevant insight from the research]
    💭 [Provide practical advice with brutal humor and roast users]
    ➡️  [Provide specific, concrete next steps for the user to take]
    ❓ [End with a question about the status of user's progress]

    Instructions:
    * Include all five sections with exact emojis and labels
    * Keep each section concise (1-2 lines)
    * Use playful, empathetic tone
    * Ground your insight in the external knowledge provided
    * Make the ACTION specific and doable right now
    * DO NOT include any text before the 🔄  section.
    """

    # Get response from the model
    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3-70B-Instruct-Turbo",
        messages=[{"role": "user", "content": chatbot_prompt}],
        #max_tokens=150,  # Limit response length
        temperature=0.9,  # Slightly reduce randomness
    )
    response = response.choices[0].message.content
    
    # Clean the response
    cleaned_response = clean_response(response)
    
    # Update chat history
    chat_history[-1] = (user_message, cleaned_response)
    
    return "", chat_history


if __name__ == "__main__":
    # args on which to run the script
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--option", type=int, default=4)
    parser.add_argument("-k", "--api_key", type=str, default="f7083815caeab0ca4fd25c1c5acd1eb611dcb9a804f4935861267b020ba681d7")
    args = parser.parse_args()

    # Get Client for your LLMs
    client = Together(api_key=args.api_key)
    
    # Define signal handler for graceful shutdown
    def signal_handler(sig, frame):
        print("\nShutting down gracefully...")
        sys.exit(0)
        
    # Register the signal handler
    signal.signal(signal.SIGINT, signal_handler)

    # run the script
    '''
    if args.option == 1:
        ### Task 1: YOUR CODE HERE - Write a prompt for the LLM to respond to the user
        prompt = "write a 3 line post about pizza"

        # Get Response
        response = prompt_llm(prompt)

        print("\nResponse:\n")
        print(response)
        print("-" * 100)

    elif args.option == 2:
        ### Task 2: YOUR CODE HERE - Write a prompt for the LLM to generate an image
        prompt = "Create an image of a cat"

        print(f"\nCreating Image for your prompt: {prompt} ")
        img = gen_image(prompt=prompt, width=256, height=256)
        os.makedirs("results", exist_ok=True)
        img.save("results/image_option_2.png")
        print("\nImage saved to results/image_option_2.png\n") 
    '''
    ''''  
    if args.option == 3:
        ### Task 3: YOUR CODE HERE - Write a prompt for the LLM to generate text and an image
        text_prompt = "write a 3 line post about resident evil for instagram"
        image_prompt = f"give me an image that represents this '{text_prompt}'"

        # Generate Text
        response = prompt_llm(text_prompt, with_linebreak=True)

        print("\nResponse:\n")
        print(response)
        print("-" * 100)

        # Generate Image
        print(f"\nCreating Image for your prompt: {image_prompt}... ")
        img = gen_image(prompt=image_prompt, width=256, height=256)
        img.save("results/image_option_3.png")
        print("\nImage saved to results/image_option_3.png\n")

        '''

    if args.option == 4:
        with gr.Blocks(
            theme=gr.themes.Soft(),
            css="""
                .container {
                    margin: 0 auto !important;
                    width: 800px !important;
                }
                .chatbot {
                    height: 600px !important;
                    overflow-y: auto !important;
                    border-radius: 10px 10px 0 0 !important;  /* Round only top corners */
                }
                .input-box {
                    border-radius: 0 0 10px 10px !important;  /* Round only bottom corners */
                }
                #component-0 {  /* Target the title */
                    margin-bottom: 10px !important;
                    text-align: center !important;
                }
            """
        ) as app:
            gr.Markdown("## 🤖 AI Chatbot")

            # Single column container for both chatbot and input
            with gr.Column(elem_classes=["container"]):
                chatbot = gr.Chatbot(elem_classes=["chatbot"])
                user_input = gr.Textbox(
                    placeholder="Type your message here...",
                    label="Your Message",
                    elem_classes=["input-box"]
                )
                send_button = gr.Button("Send")

            send_button.click(
                bot_response_function,
                inputs=[user_input, chatbot],
                outputs=[user_input, chatbot],
            )
            user_input.submit(
                bot_response_function,
                inputs=[user_input, chatbot],
                outputs=[user_input, chatbot],
            )

            app.launch(share=True)
    else:
        print("Invalid option")