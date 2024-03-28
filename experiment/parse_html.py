import pandas as pd
from bs4 import BeautifulSoup


def html_to_df(html_content):
    # Parse the HTML content
    soup = BeautifulSoup(html_content, "html.parser")

    # Find all <p> tags
    p_tags = soup.find_all("p")

    # Initialize lists to hold wave numbers and topics
    waves = []
    topics = []
    # Loop through each <p> tag
    for tag in p_tags:
        # Extract the wave number
        if not tag:
            continue
        elif not tag.find("strong"):
            continue
        wave_str = tag.find("strong").get_text()
        wave_number = wave_str.strip().split(" ")[
            -1
        ]  # Assumes the wave number is always last

        # Extract topics. Assuming there's a consistent pattern for topics after "Topics: "
        topic_text = tag.get_text()
        topic_start = topic_text.find("Topics:")  # Find the start of topics description
        if topic_start != -1:
            topic = topic_text[topic_start + len("Topics:") :].strip()
        else:  # If 'Topics:' isn't found, fallback to 'Topic:'
            topic_start = topic_text.find("Topic:")
            topic = topic_text[topic_start + len("Topic:") :].strip()

        # Append to lists
        waves.append(wave_number)
        topics.append(topic)

    # Create DataFrame
    df = pd.DataFrame({"Wave Number": waves, "Topic": topics})
    return df


# open pew_atp_html.txt
with open("pew_atp_html.txt", "r") as file:
    html_content = file.read()
    html_df = html_to_df(html_content)
    html_df.to_csv("pew_atp.csv", index=False)
