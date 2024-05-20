import openai
from openai import AzureOpenAI, BadRequestError
import time
import json
import re
# from yfinance import Ticker
import requests, os
from dotenv import load_dotenv



load_dotenv()

#-------------------------------------------------------------------------------------------------#
#----Function to get Customer Information, using Phone Number, in Json Format---------------------#
#-------------------------------------------------------------------------------------------------#

def get_customer_information(phonenumber):
    customer_info = {
        "name": "John Doe",
        "address": "123 Main St",
        "account_number": "000099998888",
        "phone_number": phonenumber
    }
    return json.dumps(customer_info)

#-------------------------------------------------------------------------------------------------#
#----Function to get Promotions, using Account Number, in Json Format-----------------------------#
#-------------------------------------------------------------------------------------------------#

def get_promotions(account_number):
    promotions = {
        "free_hulu_service": True,
        "discount_for_additional_line": "$10",
        "internet_speed_upgrade": "50 Mbps"
    }
    return json.dumps(promotions)

    

#--------------------------------------START------------------------------------------------------#
#----RAG Function getting chunks from Azure Search based on the user question---------------------#
#----Use your specific RAG function here----------------------------------------------------------#
#-------------------------------------------------------------------------------------------------#

def get_answer_from_kb(question):
    #print(f"Calling Function get_answer_from_kb with question: {question}")

    searchResult = search(question, os.getenv("AZURE_SEARCH_INDEX_NAME"), "vectorSemanticHybrid", 5, "vzw-semantic-config", "contentVector")
    combinedContent = "\n".join([result['content'] for result in searchResult]) if searchResult else ""

    return combinedContent


#------------------------Helper methods for Azure Search call----------------------#

fieldMap = {
    "id": ["id"],
    "url": ["url", "uri", "link", "document_link"],
    "filepath": ["filepath", "filename", "source"],
    "content": ["page_content"]
}
titleRegex = re.compile(r"title: (.*)\n")

def getIfString(doc, fieldName):
    try: 
        value = doc.get(fieldName)
        if isinstance(value, str) and len(value) > 0:
            return value
        return None
    except:
        return None

def get_truncated_string(string_value, max_length):
    return string_value[:max_length]

def getTitle(doc):
    max_title_length = 150
    title = getIfString(doc, 'title')
    if title:
        return get_truncated_string(title, max_title_length)
    else:
        title = getIfString(doc, 'content')
        if title: 
            titleMatch = titleRegex.search(title)
            if titleMatch:
                return get_truncated_string(titleMatch.group(1), max_title_length)
            else:
                return None
        else:
            return None

def getChunkId(doc):
    chunk_id = getIfString(doc, 'chunk_id')
    return chunk_id

def getSearchScore(doc):
    try:
        return doc['@search.score']
    except:
        return None

def getQueryList(query):
    try:
        config = json.loads(query)
        return config
    except Exception:
        return [query]

def process_search_docs_response(docs):
    outputs = []
    for doc in docs:
        formattedDoc = {}
        for fieldName in fieldMap.keys():
            for fromFieldName in fieldMap[fieldName]:
                fieldValue = getIfString(doc, fromFieldName)
                if fieldValue:
                    formattedDoc[fieldName] = doc[fromFieldName]
                    break
        formattedDoc['title'] = getTitle(doc)
        formattedDoc['chunk_id'] = getChunkId(doc)
        formattedDoc['search_score'] = getSearchScore(doc)
        outputs.append(formattedDoc)
    return outputs

def get_query_embedding(query, endpoint, api_key, api_version, embedding_model_deployment):
    request_url = f"{endpoint}/openai/deployments/{embedding_model_deployment}/embeddings?api-version={api_version}"
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key
    }
    request_payload = {
        'input': query
    }
    embedding_response = requests.post(request_url, json = request_payload, headers = headers, timeout=None)
    if embedding_response.status_code == 200:
        data_values = embedding_response.json()["data"]
        embeddings_vectors = [data_value["embedding"] for data_value in data_values]
        return embeddings_vectors
    else:
        raise Exception(f"failed to get embedding: {embedding_response.json()}")

def search_query_api(
    endpoint, 
    api_key,
    api_version, 
    index_name, 
    query_type, 
    query, 
    top_k, 
    semantic_configuration_name=None,
    vectorFields=None):
    request_url = f"{endpoint}/indexes/{index_name}/docs/search?api-version={api_version}"
    request_payload = {
        'top': top_k,
        'queryLanguage': 'en-us'
    }
    if query_type == 'simple':
        request_payload['search'] = query
        request_payload['queryType'] = query_type
    elif query_type == 'semantic':
        request_payload['search'] = query
        request_payload['queryType'] = query_type
        request_payload['semanticConfiguration'] = semantic_configuration_name
    elif query_type in ('vector', 'vectorSimpleHybrid', 'vectorSemanticHybrid'):
        embeddingModel = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL")
        if vectorFields and embeddingModel:
            query_vectors = get_query_embedding(
                query,
                os.getenv("AZURE_OPENAI_ENDPOINT"),
                os.getenv("AZURE_OPENAI_KEY"),
                os.getenv("AZURE_OPENAI_VERSION"),
                embeddingModel)
            payload_vectors = [{"value": query_vector, "fields": vectorFields, "k": top_k } for query_vector in query_vectors]
            request_payload['vectors'] = payload_vectors

        if query_type == 'vectorSimpleHybrid':
            request_payload['search'] = query
        elif query_type == 'vectorSemanticHybrid':
            request_payload['search'] = query
            request_payload['queryType'] = 'semantic'
            request_payload['semanticConfiguration'] = semantic_configuration_name
    else:
        raise Exception(f"unsupported query type: {query_type}")
    
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key
    }
    retrieved_docs = requests.post(request_url, json = request_payload, headers = headers, timeout=None)
    if retrieved_docs.status_code == 200:
        return process_search_docs_response(retrieved_docs.json()["value"])
    else:
        raise Exception(f"failed to query search index : {retrieved_docs.json()}")

def search(queries: str, indexName: str, queryType: str, topK: int, semanticConfiguration: str, vectorFields: str):
    semanticConfiguration = semanticConfiguration if semanticConfiguration != "None" else None
    vectorFields = vectorFields if vectorFields != "None" else None
                      
    # Do search.
    allOutputs = [search_query_api(
        os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT"), 
        os.getenv("AZURE_SEARCH_ADMIN_KEY"),
        os.getenv("AZURE_SEARCH_VERSION"), 
        indexName,
        queryType,
        query, 
        topK, 
        semanticConfiguration,
        vectorFields) for query in getQueryList(queries)]

    includedOutputs = []
    while allOutputs and len(includedOutputs) < topK:
        for output in list(allOutputs):
            if len(output) == 0:
                allOutputs.remove(output)
                continue
            value = output.pop(0)
            if value not in includedOutputs:
                includedOutputs.append(value)
                if len(includedOutputs) >= topK:
                    break
    return includedOutputs

#--------------------------------------RAG Function END------------------------------------------------------#



#-------------------------------------------------------------------------------------------------#
#----Function to make Assistants API call---------------------------------------------------------#
#-------------------------------------------------------------------------------------------------#

def process_llm_request(question: str):


    tools_list = [
        {
            "type": "function",
            "function": {
                "name": "get_customer_information",
                "description": "Get the customer information based on their phone number",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "phonenumber": {
                            "type": "string",
                            "description": "Customer phone number",
                        },
                    },
                    "required": ["phonenumber"]
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_promotions",
                "description": "Get the sales promotions for the customer based on their account number",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "account_number": {
                            "type": "string",
                            "description": "Customer account number",
                        },
                    },
                    "required": ["account_number"]
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_answer_from_kb",
                "description": "Answer customer query using the data from knowledge base",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "customer query to be answered using the knowledge base",
                        },
                    },
                    "required": ["question"]
                },
            },
        }
    ]

    # Initialize OPENAI client
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version=os.getenv("AZURE_OPENAI_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )

    # Step 1: Create an Assistant
    assistant = client.beta.assistants.create(
        name="Call Center Chat Assistant",
        instructions="You are a personal  Chat Assistant",
        model="gpt-35-turbo-16k",
        tools=tools_list,
    )

    # Step 2: Create a Thread
    thread = client.beta.threads.create()

    # Step 3: Add a Message to a Thread
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=question
    )

    # Step 4: Run the Assistant
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id,
        instructions="Please address the user as Bot."
    )

    #print(run.model_dump_json(indent=4))

    # Define a dispatch table
    function_dispatch_table = {
        "get_promotions": get_promotions,
        "get_customer_information": get_customer_information,
        "get_answer_from_kb": get_answer_from_kb
    }

    run_result = None
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
                if role == 'assistant': # If the role is assistant, print the assistant response
                    return content      # Save the assistant response and return it
            break
        elif run_status.status == 'requires_action':
            #print("Calling the required functions...")
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


#--------------------------------------END------------------------------------------------------#


#--------------------------------------Main loop------------------------------------------------------#
while True:
    # Get user input and display text in green color
    user_input = input("\033[92mEnter user question: \033[0m")

    if user_input == '':
        user_input = """Can you please provide me customer information for phone number 123-456-7890,
        promotions available for the same customer,
        if customer need to qualify an address for 5G service?"""
        print("Using the default question:")
        print(user_input)

    if user_input == 'exit': # Exit the loop if user enters 'exit' or CTRL-C
        break

    response = process_llm_request(user_input)
    print(response)
print("Goodbye!")