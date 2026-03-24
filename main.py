from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter

load_dotenv()

model = ChatOpenRouter(
    model="openrouter/free",
)


def main():
    messages = [
        (
            "system",
            "Du bist ein hilfreicher Assistent. Antworte auf Deutsch.",
        ),
        ("human", "Was ist ein RAG-System? Erkläre es in einem Satz."),
    ]
    ai_msg = model.invoke(messages)
    print(ai_msg.content)


if __name__ == "__main__":
    main()