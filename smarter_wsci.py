from pathlib import Path
from ollama import chat
import json



question = """
I changed my university password this morning.
Now my Windows laptop won't connect to campus Wi-Fi,
but my phone still works.
"""

## WRITE ##
service_status = {
    "wifi": "operational"
}

state = {
    "problem": question,
    "wi_fi status": "operational",
    "wi-fi_check": True
}

with open("state.json", "w") as file:
    json.dump(
        state,
        file,
        indent=2
    )

with open("state.json", "r") as file:
    state = json.load(file)

print(state)


## SELECT CONTEXT FILES BASED ON QUESTION
## Create the function that takes the student's question, takes some keywords and chooses the relevant files from the knowledge base. Return a list of the selected files.
## For example, if the question has the kyeword "print" or "printer", then the function should return the file "knowledge/printer_setup.txt" in a list.
def select_context(question):
    keywords = {
        "wi-fi": "knowledge/wifi_setup.txt",
        "wifi": "knowledge/wifi_setup.txt",
        "password": "knowledge/password_changes.txt",
        "email": "knowledge/email_setup.txt",
        "vpn": "knowledge/vpn.txt",
        "print": "knowledge/printing.txt",
        "projector": "knowledge/classroom_projectors.txt",
        "display": "knowledge/classroom_projectors.txt",
        "status": "knowledge/service_status.txt"
    }

    question = question.lower()
    selected_files = []

    for keyword in keywords:
        if keyword in question:
            file = keywords[keyword]
            if file not in selected_files:
                selected_files.append(file)

    return selected_files


selected_files = select_context(question)

print(
    "Selected files:",
    selected_files
)

## READ SELECTED FILES and add their contents to the context variable.
context = ""

for file in selected_files:
    context += Path(file).read_text()
    context += "\n\n"

print(
    "Context characters:",
    len(context)
)

## 
## COMPRESS CONTEXT
## Add logic to compress the context from above by calling Qwen with "context" and the "question" as the parameter
## The response from Qwen should be the compressed context. Store it in a variable called "compressed_context" 

def compress_context(context, question):
    response = chat(
        model="qwen3:8b",
        messages=[
            {
                "role": "system",
                "content": "You are a university IT support assistant. From the knowledge base below, keep only the information that is needed to solve the student's problem and leave everything else out. Answer with the compressed knowledge base only."
            },
            {
                "role": "user",
                "content": "Student problem:\n" + question + "\nKnowledge base:\n" + context
            }
        ]
    )

    return response.message.content


compressed_context = compress_context(context, question)


## Print the length of the compressed context
print(len(compressed_context))

## Now, call Qwen again with the compressed context and the student's question. Store the response in a variable called "response" and print the response from Qwen.
## Ensure the model produces a structured output 

response = chat(
    model="qwen3:8b",
    messages=[
        {
            "role": "system",
            "content": "You are a university IT support assistant. Solve the student's problem using only the information below. Answer with JSON only and use exactly these keys: \"device\" (the device with the problem), \"cause\" (the most likely cause, in one sentence) and \"steps\" (the steps the student should follow, as a list of strings).\n\n" + compressed_context
        },
        {
            "role": "user",
            "content": question
        }
    ],
    format="json"
)

print(response.message.content)

answer = json.loads(response.message.content)


## WRITE the above output in an artifact called "state"

## The artifact has separate parts. This way the next prompt can use only the part it needs.

diagnostic_context = {
    "problem": question,
    "device": answer["device"],
    "wifi_status": service_status["wifi"],
    "cause": answer["cause"],
    "steps": answer["steps"]
}

report_context = {
    "total_wifi_cases": 37,
    "resolved_cases": 29,
    "unresolved_cases": 8
}

state = {
    "diagnostic": diagnostic_context,
    "report": report_context
}

with open("state.json", "w") as file:
    json.dump(
        state,
        file,
        indent=2
    )


## Update the rest of the code so that it uses the "state" artifact as part of the context. 
## It is important to ensure that the model uses only the relevant parts from the "state" artifact and not the entire artifact.
## For this, you may have to think of a good structure for the "state" artifact and how to use it in the context.

## ISOLATE
## Qwen classifies the problem first, and the answer tells the program which part of the state to use.

classification = chat(
    model="qwen3:8b",
    messages=[
        {
            "role": "system",
            "content": "Classify the student's request. Answer with JSON only: use {\"task\": \"diagnostic\"} if the student is asking for help with a device or a service, or {\"task\": \"report\"} if the student is asking for statistics about earlier cases."
        },
        {
            "role": "user",
            "content": question
        }
    ],
    format="json"
)

task = json.loads(classification.message.content)["task"]

print(
    "Task:",
    task
)

with open("state.json", "r") as file:
    state = json.load(file)

if task == "report":
    state_part = state["report"]
else:
    state_part = state["diagnostic"]

## Only the part that was selected is put into the context, not the whole state artifact.
response = chat(
    model="qwen3:8b",
    messages=[
        {
            "role": "system",
            "content": "You are a university IT support assistant. Answer the student using only the state below.\n\n" + json.dumps(state_part, indent=2)
        },
        {
            "role": "user",
            "content": question
        }
    ]
)

print(response.message.content)
