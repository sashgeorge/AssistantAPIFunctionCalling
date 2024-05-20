import openai
from openai import AzureOpenAI, BadRequestError
import time
import json
from yfinance import Ticker
import requests, os
from dotenv import load_dotenv

load_dotenv()


#-------------------------------------------------------------------------------------------------#
#function description: get_stock_price
#-------------------------------------------------------------------------------------------------#

def get_stock_price(symbol: str) -> float:
    stock = Ticker(symbol)
    price = stock.history(period="1d")['Close'].iloc[-1]
    return price


#-------------------------------------------------------------------------------------------------#
#function description: get_latest_company_news
#-------------------------------------------------------------------------------------------------#
def get_latest_company_news(company_name):
    response = call_bing(company_name)
    return response['news']


def call_bing(query):
    subscription_key = os.getenv("BING_SEARCH_KEY")
    endpoint = os.getenv('BING_SEARCH_ENDPOINT') 

    # Construct a request
    mkt = 'en-US'
    params = { 'q': query, 'mkt': mkt, }
    headers = { 'Ocp-Apim-Subscription-Key': subscription_key }


    # Call the API
    response = requests.get(endpoint, headers=headers, params=params)
    #response.raise_for_status()
    
    if response.status_code == 200:
        return response.json()
    else:
        return []


#-------------------------------------------------------------------------------------------------#
#function description: usd_to_gbp
#-------------------------------------------------------------------------------------------------#

def usd_to_gbp(usd_amount):
    url = "https://api.exchangerate-api.com/v4/latest/USD"
    try:
        response = requests.get(url)
        data = response.json()
        gbp_rate = data['rates']['GBP']
        return usd_amount * gbp_rate
    except Exception as e:
        return f"Error: {e}"


#-------------------------------------------------------------------------------------------------#
#function description: process_llm_request
#-------------------------------------------------------------------------------------------------#

def process_llm_request(question: str):

    tools_list = [
        {
            "type": "function",
            "function": {
                "name": "get_stock_price",
                "description": "Retrieve the latest closing price of a stock using its ticker symbol",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "The ticker symbol of the stock"
                        }
                    },
                    "required": ["symbol"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_latest_company_news",
                "description": "Fetches the latest news articles related to a specified company",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "company_name": {
                            "type": "string",
                            "description": "The name of the company"
                        }
                    },
                    "required": ["company_name"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "usd_to_gbp",
                "description": "Converts an amount in USD to GBP using the current exchange rate",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "usd_amount": {
                            "type": "number",
                            "description": "The amount in USD to be converted"
                        }
                    },
                    "required": ["usd_amount"]
                }
            }
        }
    ]

    # Initialize the client
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version=os.getenv("AZURE_OPENAI_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )

    # Step 1: Create an Assistant
    assistant = client.beta.assistants.create(
        name="Data Analyst Assistant",
        instructions="You are a personal Data Analyst Assistant",
        model="gpt-35-turbo-16k",
        tools=tools_list,
    )

    # Step 2: Create a Thread
    thread = client.beta.threads.create()

    # Step 3: Add a Message to a Thread
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content= question
    )

    # Step 4: Run the Assistant
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id,
        instructions="Please address the user as Bot."
    )

    # print(run.model_dump_json(indent=4))

    # Define a dispatch table
    function_dispatch_table = {
        "get_stock_price": get_stock_price,
        "get_latest_company_news": get_latest_company_news,
        "usd_to_gbp": usd_to_gbp
    }

    while True:
        # Wait for 5 seconds
        time.sleep(5)

        # Retrieve the run status
        run_status = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id
        )
        #print(run_status.model_dump_json(indent=4))

        # If run is completed, get messages
        if run_status.status == 'completed':
            messages = client.beta.threads.messages.list(
                thread_id=thread.id
            )

            # Loop through messages and print content based on role
            for msg in messages.data:
                role = msg.role
                content = msg.content[0].text.value
                #print(f"{role.capitalize()}: {content}")
                if role == 'assistant': 
                    return content
            break
        elif run_status.status == 'requires_action':
            required_actions = run_status.required_action.submit_tool_outputs.model_dump()
            #print(required_actions)
            tools_output = []

            for action in required_actions["tool_calls"]:
                func_name = action["function"]["name"]
                arguments = json.loads(action["function"]["arguments"])

                print(f"\033[93mFunction: {func_name}, Arguments: {arguments}\033[0m")

                func = function_dispatch_table.get(func_name)
                if func:
                    result = func(**arguments)
                    # Ensure the output is a JSON string
                    output = json.dumps(result) if not isinstance(result, str) else result
                    tools_output.append({
                        "tool_call_id": action["id"],
                        "output": output
                    })
                else:
                    print(f"Function {func_name} not found")

            # Submit the tool outputs to Assistant API
            client.beta.threads.runs.submit_tool_outputs(
                thread_id=thread.id,
                run_id=run.id,
                tool_outputs=tools_output
            )


        else:
            print("Waiting for the Assistant to process...")
            time.sleep(5)

while True:
    # Get user input and display text in green color

    user_input = input("\033[92mEnter your question: \033[0m")
    
    #Use default question if user input is empty
    if user_input == '':
        user_input = """Can you please provide me stock price,
        stock price in GBP,
        and the latest company news of Microsoft?"""
        print("Using the default question:")
        print(user_input)

    if user_input == 'exit':
        break

    response = process_llm_request(user_input)
    print(response)
print("Goodbye!")