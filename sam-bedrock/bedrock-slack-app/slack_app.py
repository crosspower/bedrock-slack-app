import os
import boto3
import re
import time
import json
import logging

from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_bolt.adapter.aws_lambda import SlackRequestHandler
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Bedrock
from langchain_community.retrievers import AmazonKnowledgeBasesRetriever 
from langchain_core.runnables import RunnablePassthrough

CHAT_UPDATE_INTERVAL_SEC = 1

load_dotenv()

SlackRequestHandler.clear_all_log_handlers()
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO
)

logger = logging.getLogger(__name__)

app = App(
    signing_secret=os.environ["SLACK_SIGNING_SECRET"],
    token=os.environ["SLACK_BOT_TOKEN"],
    process_before_response=True
)

# ================================================================
# Retrieve用のプロンプトの定義
prompt_pre = PromptTemplate.from_template("""
あなたはquestionから、検索ツールへの入力となる検索キーワードを考えます。
questionに後続処理への指示（例：「説明して」「要約して」）が含まれる場合は取り除きます。
検索キーワードは文章では無く簡潔な単語で指定します。
検索キーワードは複数の単語を受け付ける事が出来ます。
検索キーワードは日本語が標準ですが、ユーザー問い合わせに含まれている英単語はそのまま使用してください。
回答形式は文字列です。
<question>{question}</question>
""")
# ================================================================

# ================================================================
# 回答生成用のプロンプトの定義
prompt_main = PromptTemplate.from_template("""
あなたはcontextを参考に、questionに回答します。
<context>{context}</context>
<question>{question}</question>
""")
# ================================================================

# LLMの定義
LLM = Bedrock(
    model_id="anthropic.claude-v2:1",
    model_kwargs={"max_tokens_to_sample": 1000},
)

# Retriever(Kendra)の定義（Kendra Index ID、言語、取得件数）
retriever = AmazonKnowledgeBasesRetriever(
    knowledge_base_id=os.environ["KNOWLEDGE_BASE_ID"],
    retrieval_config={
        "vectorSearchConfiguration": {
            "numberOfResults": 4
        }
    },
)
# chainの定義
chain = (
    {"context": prompt_pre | LLM | retriever, "question":  RunnablePassthrough()}
    | prompt_main 
    | LLM
)
# chain = {"context": retriever, "question": RunnablePassthrough()} | prompt_main | LLM

# @app.event("app_mention")
def handle_mention(event, say):
    channel = event["channel"]
    thread_ts = event["ts"]
    messasge = re.sub("<@.*>", "", event["text"])
    
    # いったん書き込む
    result = say("\n\nTyping...", thread_ts=thread_ts)
    
    for_update_ts = result["ts"]
    
    
    # chainの実行
    output_text = ""
    last_send_time = time.time()
    interval = CHAT_UPDATE_INTERVAL_SEC
    update_count = 0
    for chunk in chain.stream({"question": messasge}):
        output_text += chunk
        now = time.time()
        if now - last_send_time > interval:
            last_send_time = now
            update_count += 1
            
            app.client.chat_update(
                channel=channel, ts=for_update_ts, text=f"{output_text}..."
            )
            
            # update_countが現在の更新間隔 x 10 より多くなるたびに更新間隔を2倍にする
            if update_count / 10 > interval:
                interval = interval * 2
    # answer = chain.invoke({"question": messasge})
    # print(answer)

    # response = client.retrieve_and_generate(
    #     input={
    #         'text': messasge
    #     },
    #     retrieveAndGenerateConfiguration={
    #         'type': 'KNOWLEDGE_BASE',
    #         'knowledgeBaseConfiguration': {
    #             'knowledgeBaseId': knowledgebaseId,
    #             'modelArn': modelArn,
    #         },
    #     },
    # )
    
    # output_text = response['output']['text']

    
    message_context = "生成AIによって生成される情報は不正確または不適切な場合がありますが、当社の見解を述べるものではありません。"
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
    
# def lambda_handler(event, context):
#     logger.info("handler called")
#     header = event["headers"]
#     logger.info(json.dumps(header))
    
#     if "x-slack-retry-num" in header:
#         logger.info("SKIP > x-slack-retry-num: %s", header["x-slack-retry-num"])
#         return 200
        
        
#     # AWS Lambda環境のリクエスト情報をappが処理できるように変換するアダプター
#     slack_handler = SlackRequestHandler(app=app)
#     # 応答はそのまま AWS Lambdaの戻り値として返す
#     return slack_handler.handle(event, context)