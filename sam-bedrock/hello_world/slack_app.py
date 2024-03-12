import os
import boto3
import re
from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

CHAT_UPDATE_INTERVAL_SEC = 1
region = "us-east-1"
modelId = "anthropic.claude-v2:1"
knowledgebaseId = "YQMHXBRSRU"
modelArn = f'arn:aws:bedrock:{region}::foundation-model/{modelId}'

session = boto3.Session(region_name=region)
client = session.client(service_name='bedrock-agent-runtime')


load_dotenv()

app = App(
    signing_secret=os.environ["SLACK_SIGNING_SECRET"],
    token=os.environ["SLACK_BOT_TOKEN"],
    process_before_response=True
)


# @app.event("app_mention")
def handle_mention(event, say):
    channel = event["channel"]
    thread_ts = event["ts"]
    messasge = re.sub("<@.*>", "", event["text"])
    
    # いったん書き込む
    result = say("\n\nTyping...", thread_ts=thread_ts)
    
    for_update_ts = result["ts"]
    
    response = client.retrieve_and_generate(
        input={
            'text': messasge
        },
        retrieveAndGenerateConfiguration={
            'type': 'KNOWLEDGE_BASE',
            'knowledgeBaseConfiguration': {
                'knowledgeBaseId': knowledgebaseId,
                'modelArn': modelArn,
            },
        },
    )
    
    output_text = response['output']['text']

    
    message_context = "OpenAI APIで生成される情報は不正確または不適切な場合がありますが、当社の見解を述べるものではありません。"
    message_blocks = [
        {"type": "section", "text": {"type": "mrkdwn", "text": output_text}},
        {"type": "divider"},
        {
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": message_context}],
        },
    ]
    
    app.client.chat_update(
        channel = channel,
        ts=for_update_ts,
        text=output_text,
        blocks=message_blocks
    )
    # say(thread_ts=thread_ts, text=output_text)

def just_ack(ack):
    ack()
    
app.event("app_mention")(ack=just_ack, lazy=[handle_mention])

if __name__ == "__main__":
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()