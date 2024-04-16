import pandas as pd
import time

# Assuming you have already defined classes and functions mentioned in your code snippet

# Function to extract detailed summary and metadata for each video URL
def extract_summary_and_metadata(video_url):
    loader = YoutubeLoader.from_youtube_url(video_url, add_video_info=True)
    result = loader.load()

    metadata = result[0].metadata
    metadata['source'] = 'https://www.youtube.com/watch?v=' + metadata['source']
    metadata['type'] = 'Youtube'
    del metadata['description']
    del metadata['thumbnail_url']

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100000, chunk_overlap=500)
    texts = text_splitter.split_documents(result)

    # Assuming the defined classes and functions in your code snippet
    llm = ChatAnthropic(temperature=0, max_tokens=4000, model_name="claude-3-haiku-20240307", anthropic_api_key="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")

    prompt_template = """
    <Task>
    Generate a highly detailed summary of a YouTube video transcript focused on cryptocurrencies.
    </Task>
    <Inputs>
    {text}
    </Inputs>
    <Instructions>
    Write a very detailed summary of a youtube video transcript. Below are the instructions:
    - Read the provided YouTube video transcript thoroughly.
    - Extract any financial advice mentioned in the transcript, along with the reasoning, arguments and claims behind it.
    - Identify and list all named entities and cryptocurrencies mentioned in the transcript.
    - Assess the overall sentiment of the transcript and determine whether it is bullish or bearish.
    </Instructions>
    DETAILED SUMMARY:"""
    prompt = PromptTemplate.from_template(prompt_template)

    refine_template = (
        "Your job is to produce a final summary\n"
        "We have provided an existing summary up to a certain point: {existing_answer}\n"
        "We have the opportunity to refine the existing summary"
        "(only if needed) with some more context below.\n"
        "------------\n"
        "{text}\n"
        "------------\n"
        "Given the new context, refine the original detailed summary"
        "If the context isn't useful, return the original summary."
    )
    refine_prompt = PromptTemplate.from_template(refine_template)
    chain = load_summarize_chain(
        llm=llm,
        chain_type="refine",
        question_prompt=prompt,
        refine_prompt=refine_prompt,
        return_intermediate_steps=True,
        input_key="input_documents",
        output_key="output_text"
    )
    result = chain({"input_documents": texts}, return_only_outputs=True)

    return result, metadata


# DataFrame with video URLs
# Assuming you have a DataFrame named latest_videos with columns: Channel_name, Video_title, Video_url
# Assuming you want to add the detailed summary and metadata in new columns

# Define new columns for detailed summary and metadata
latest_videos['Detailed_Summary'] = ""
latest_videos['Video_Source'] = ""
latest_videos['Video_Type'] = ""

# Iterate through each row in the DataFrame
for index, row in latest_videos.iterrows():
    video_url = row['Video_url']

    # Extract summary and metadata
    detailed_summary, metadata = extract_summary_and_metadata(video_url)

    # Update DataFrame with summary and metadata
    latest_videos.at[index, 'Detailed_Summary'] = detailed_summary
    latest_videos.at[index, 'Video_Source'] = metadata['source']
    latest_videos.at[index, 'Video_Type'] = metadata['type']

    # Delay of 60 seconds
    time.sleep(60)

# Now latest_videos DataFrame contains detailed summaries and metadata for each video URL
