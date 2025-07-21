import socket
import time
import json
from random import choice

sample_data = [
    {"text": "Stock markets are rallying due to strong earnings.", "label": "positive"},
    {"text": "Investors fear recession amid rising inflation.", "label": "negative"},
    {"text": "The economy shows mixed signals this quarter.", "label": "neutral"},

    {"text": "Tech giants report record-breaking profits.", "label": "positive"},
    {"text": "Global markets tumble over geopolitical tensions.", "label": "negative"},
    {"text": "The Federal Reserve maintains interest rates.", "label": "neutral"},

    {"text": "Job growth exceeds expectations in the latest report.", "label": "positive"},
    {"text": "Unemployment claims rise sharply.", "label": "negative"},
    {"text": "Central bank policy remains unchanged.", "label": "neutral"},

    {"text": "Consumer confidence hits a five-year high.", "label": "positive"},
    {"text": "Retail sales drop as inflation bites.", "label": "negative"},
    {"text": "Market reacts cautiously to new trade deal.", "label": "neutral"},

    {"text": "Energy stocks soar with rising oil prices.", "label": "positive"},
    {"text": "Layoffs increase in the tech industry.", "label": "negative"},
    {"text": "Analysts await quarterly earnings results.", "label": "neutral"},

    {"text": "Housing market rebounds after months of decline.", "label": "positive"},
    {"text": "Natural disasters disrupt supply chains.", "label": "negative"},
    {"text": "Bond yields remain flat amid global uncertainty.", "label": "neutral"},

    {"text": "Investor optimism grows with signs of recovery.", "label": "positive"},
    {"text": "Banking sector under pressure after earnings miss.", "label": "negative"},
]

def start_news_socket():
    host = "localhost"
    port = 9999
    s = socket.socket()
    s.bind((host, port))
    s.listen(1)
    print(f"Socket listening on {host}:{port}")

    conn, addr = s.accept()
    print(f"Connection from: {addr}")

    while True:
        article = choice(sample_data)
        message = json.dumps(article)
        conn.send((message + "\n").encode())
        time.sleep(2)  # simulate delay

if __name__ == "__main__":
    start_news_socket()
