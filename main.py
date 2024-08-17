import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import yt_dlp
from dotenv import load_dotenv
from openai import OpenAI

from deepgram import (
    DeepgramClient,
    PrerecordedOptions,
    FileSource,
)
from deepgram_captions import DeepgramConverter, srt

app = FastAPI()
API_KEY = "6db495d0cf32a30d7a675e9de79d0c2e6ba4356e"
client = OpenAI(api_key="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")

class URLItem(BaseModel):
    url: str

class FilePath(BaseModel):
    path: str



@app.post("/download_audio/")
async def fetch_url_content(item: URLItem):
    try:
        url = item.url
        print(url)
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'ffmpeg_location': r"C:\ffmpeg\bin",
            'outtmpl': f'/%(title)s.%(ext)s',
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return {"audio_path": f"{ydl.extract_info(url, download=False)['title']}.wav"}

    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/delete_files/")
async def delete_audio(file: FilePath):
    try:
        audio_path = file.audio_path
        caption_path = file.caption_path
        if os.path.exists(audio_path) and os.path.exists(caption_path):
            os.remove(audio_path)
            os.remove(caption_path)
            return {"message": "Files deleted successfully."}
        else:
            raise HTTPException(status_code=404, detail="Files not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/get_transcript/")
async def get_transcript(file: FilePath):
    try:
        if os.path.exists(file.path):
            audio = file.path
            deepgram = DeepgramClient(API_KEY)

            with open(audio, "rb") as file:
                buffer_data = file.read()

            payload: FileSource = {
                "buffer": buffer_data,
            }

            # STEP 2: Configure Deepgram options for audio analysis
            options = PrerecordedOptions(
                model="nova-2",
                smart_format=True,
            )

            # STEP 3: Call the transcribe_file method with the text payload and options
            response = deepgram.listen.prerecorded.v("1").transcribe_file(payload, options)
            transcription = DeepgramConverter(response)

            # for SRT captions
            captions = srt(transcription)
            caption_path = f"{audio}.srt"
            with open(caption_path, "w") as file:
                file.write(captions)
            return {"caption_path": caption_path}
        else:
            raise HTTPException(status_code=404, detail=f"File '{file.path}' not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

class Item(BaseModel):
    path: str
    links: str
    context: str

@app.post("/get_description/")
async def get_description(item: Item):
    try:
        if os.path.exists(item.path):
            with open(item.path, "r") as file:
                transcript = file.read()
            links = item.links
            context = item.context
            completion = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": """
                        You are a highly capable and creative language model tasked with generating concise and SEO-optimized descriptions for YouTube videos. The user will provide a video transcript, and your role is to:

                        1. Extract the key points, main ideas, and important details from the transcript.
                        2. Craft a clear, engaging, and informative summary that is concise yet captures the essence of the video.
                        3. Include any provided links in the description, ensuring they are presented clearly and are relevant to the content.
                        4. Add SEO-optimized tags that are relevant to the video content to help improve discoverability.
                        5. The description should be visually appealing, including relevant emojis where appropriate to make it engaging.
                        6. If the user provides context at the beginning, incorporate it into the description.

                        The final output should be in a format typically used for YouTube video descriptions, including the summary, links, and tags.
                        """},
                    {"role": "user",
                     "content": f"Context (if any): {context}\nTranscript: {transcript}\nLinks: {links}"}
                ]
            )
            return {"data": completion.choices[0].message}
        else:
            raise HTTPException(status_code=404, detail=f"File '{item.path}' not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

class Chapter(BaseModel):
    path: str

@app.post("/get_chapters/")
async def get_chapters(item: Chapter):
    try:
        if os.path.exists(item.path):
            with open(item.path, "r") as file:
                transcript = file.read()
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": """
                        Given the following transcript of a YouTube video, identify and create a list of chapters with their timestamps. The chapters should be formatted in a clear, concise manner, with each chapter title reflecting the key topic or section covered. Use the following format:
                        0:00 Intro
                        1:23 Chapter Title 1
                        3:45 Chapter Title 2
                        ...
                        Please ensure that the timestamps are accurate and that the chapter titles are descriptive but concise. If a specific section does not have a clear title from the transcript, summarize the main topic or activity discussed.
            """},
                    {"role": "user", "content": f"Transcript: {transcript}"}
                ]
            )
            return {"data": completion.choices[0].message}
        else:
            raise HTTPException(status_code=404, detail=f"File '{item.path}' not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

class SMItem(BaseModel):
    path: str
    context: str

@app.post("/get_social_media/")
async def get_social_media(item: SMItem):
    try:
        if os.path.exists(item.path):
            with open(item.path, "r") as file:
                transcript = file.read()
            completion = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": """
                        You are a highly creative language model specializing in crafting engaging social media content. The user will provide a transcript of a video, and your task is to:
        
                        1. Extract the key points, main ideas, and essential details from the transcript.
                        2. Craft a concise, catchy, and engaging social media caption that captures the essence of the video and encourages interaction.
                        3. Include relevant and SEO-optimized hashtags that are popular within the context of the video content to improve discoverability.
                        4. Incorporate appropriate emojis to make the caption more visually appealing and relatable to the target audience.
        
                        The final output should be formatted for social media platforms such as Instagram, Twitter, or Facebook.
        
                        The output should be structured as follows:
        
                        - Caption: "Your catchy caption here 🚀"
                        - Tags: #exampletag1 #exampletag2 #exampletag3
                        """},
                    {"role": "user", "content": f"Context (if any): {item.context}\nTranscript: {transcript}"}
                ]
            )
            return {"data": completion.choices[0].message}
        else:
            raise HTTPException(status_code=404, detail=f"File '{item.path}' not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
