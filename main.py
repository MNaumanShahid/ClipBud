import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import yt_dlp
from dotenv import load_dotenv
from openai import OpenAI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uuid
from deepgram import (
    DeepgramClient,
    PrerecordedOptions,
    FileSource,
)
from deepgram_captions import DeepgramConverter, srt

app = FastAPI()
API_KEY = "6db495d0cf32a30d7a675e9de79d0c2e6ba4356e"
client = OpenAI()

origins = [
    "http://localhost:5173"
    # Add more allowed origins as needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows specific origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

class URLItem(BaseModel):
    url: str

class FilePath(BaseModel):
    path: str

class DelPath(BaseModel):
    audio_path: str
    caption_path: str

@app.post("/make_transcript/")
async def fetch_url_content(item: URLItem):
    try:
        url = item.url
        if url == "https://www.youtube.com/watch?v=j8IYsQ6QVp8":
            return {"audio_path": "audio1.wav", "caption_path": "audio1.srt"}
        elif url == "https://www.youtube.com/watch?v=06kJXhOZhLU":
            return {"audio_path": "audio2.wav", "caption_path": "audio2.srt"}
        elif url == "https://www.youtube.com/watch?v=vJEbP2Vdq2U":
            return {"audio_path": "audio3.wav", "caption_path": "audio3.srt"}
        uid = str(uuid.uuid4())
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'ffmpeg_location': r"C:\ffmpeg\bin",
            'outtmpl': f'{uid}.%(ext)s',
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        audio_path = f"{uid}.wav"

        deepgram = DeepgramClient(API_KEY)

        with open(audio_path, "rb") as file:
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
        caption_path = audio_path.replace(".wav", ".srt")
        with open(caption_path, "w") as file:
            file.write(captions)

        return {"audio_path": audio_path, "caption_path": caption_path}

    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/get_transcript/")
async def get_audio(file: FilePath):
    try:
        if os.path.exists(file.path):
            audio = file.path
            return FileResponse(audio, media_type="text/plain", filename="subtitle_file.srt")
        else:
            raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/delete_files/")
async def delete_audio(file: DelPath):
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

# @app.post("/get_transcript/")
# async def get_transcript(file: FilePath):
#     try:
#         if os.path.exists(file.path):
#             audio = file.path
#             deepgram = DeepgramClient(API_KEY)
#
#             with open(audio, "rb") as file:
#                 buffer_data = file.read()
#
#             payload: FileSource = {
#                 "buffer": buffer_data,
#             }
#
#             # STEP 2: Configure Deepgram options for audio analysis
#             options = PrerecordedOptions(
#                 model="nova-2",
#                 smart_format=True,
#             )
#
#             # STEP 3: Call the transcribe_file method with the text payload and options
#             response = deepgram.listen.prerecorded.v("1").transcribe_file(payload, options)
#             transcription = DeepgramConverter(response)
#
#             # for SRT captions
#             captions = srt(transcription)
#             caption_path = f"{audio}.srt"
#             with open(caption_path, "w") as file:
#                 file.write(captions)
#             return {"caption_path": caption_path}
#         else:
#             raise HTTPException(status_code=404, detail=f"File '{file.path}' not found.")
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

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
                model="gpt-4o",
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
                model="gpt-4o",
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

class Reels(BaseModel):
    path: str

@app.post("/get_reels/")
async def get_reels(item: Reels):
    try:
        if os.path.exists(item.path):
            with open(item.path, "r") as file:
                transcript = file.read()
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": """
                        You are a highly skilled language model specialized in content creation and optimization for social media platforms. The user will provide a transcript of a video, and your task is to:

                        1. Analyze the transcript to identify the most engaging, impactful, and content-rich segments that would be suitable for creating reels or shorts.
                        2. Focus on moments that are not only attention-grabbing but also provide a complete thought, story, or demonstration, ensuring that the segments are coherent and do not end abruptly.
                        3. Ensure that each selected segment is of sufficient length, typically ranging between 15 to 60 seconds, to provide valuable and meaningful content for the audience.
                        4. Provide the timestamps for these segments in the format `[start time] - [end time]`.
                        5. Avoid selecting segments that are too short or that cut off important information, ensuring that each clip feels complete and engaging on its own.
                        6. Select at most 4 segments.

                        The final output should only include the timestamps in the following format:

                        ```
                        [start time] - [end time]
                        [start time] - [end time]
                        [start time] - [end time]
                        ```

                        No additional text or descriptions are needed; just the timestamps.
                        """},
                    {"role": "user", "content": f"Transcript: {transcript}"}
                ]
            )
            return {"data": completion.choices[0].message}
        else:
            raise HTTPException(status_code=404, detail=f"File '{item.path}' not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

class SummaryItem(BaseModel):
    path: str

@app.post("/get_summary/")
async def get_summary(item: SummaryItem):
    try:
        if os.path.exists(item.path):
            with open(item.path, "r") as file:
                transcript = file.read()
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": """
                        You are a skilled language model tasked with creating informative and concise summaries of video content. The user will provide a transcript of a video, and your role is to:

                        1. Extract the key points, main ideas, and essential details from the transcript.
                        2. Craft a clear, engaging, and informative summary that gives the viewer an overall picture of what the video is about.
                        3. Ensure the summary covers the main topics and any important insights, so that the viewer understands the core message of the video without needing to watch it in full.
                        4. The tone should be neutral, informative, and accessible, suitable for a broad audience.

                        The final summary should be concise yet comprehensive, providing enough detail to convey the video's content effectively.

                        The output should be structured as follows:

                        - Summary: "Your summary here"
                        """},
                    {"role": "user", "content": f"Transcript: {transcript}"}
                ]
            )
            return {"data": completion.choices[0].message}
        else:
            raise HTTPException(status_code=404, detail=f"File '{item.path}' not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

class HighlightItem(BaseModel):
    path: str

@app.post("/get_highlight/")
async def get_highlight(item: HighlightItem):
    try:
        if os.path.exists(item.path):
            with open(item.path, "r") as file:
                transcript = file.read()
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": """
                        You are an expert in summarizing and extracting key highlights from video content. The user will provide a transcript of a video, and your task is to:

                        1. Analyze the transcript to identify the most important and impactful parts of the video that serve as highlights.
                        2. For each highlight, provide the start time and stop time in the format `[start time] - [stop time]`.
                        3. After listing the time ranges, provide a brief description of what each highlight covers, explaining its significance or the key point it conveys.
                        4. Ensure the format is clear and consistent so that the response can be easily parsed.

                        The final output should be structured as follows:

                        ```
                        Highlights
                        [start time] - [stop time]: Description of the highlight
                        [start time] - [stop time]: Description of the highlight
                        [start time] - [stop time]: Description of the highlight
                        ```

                        Make sure the highlights cover the main points or the most engaging moments of the video.
                        """},
                    {"role": "user", "content": f"Transcript: {transcript}"}
                ]
            )
            return {"data": completion.choices[0].message}
        else:
            raise HTTPException(status_code=404, detail=f"File '{item.path}' not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

class ThumbnailItem(BaseModel):
    path: str

@app.post("/get_thumbnail/")
async def get_thumbnail(item: ThumbnailItem):
    try:
        if os.path.exists(item.path):
            with open(item.path, "r") as file:
                transcript = file.read()
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": """
                        You are a creative and detail-oriented language model with expertise in generating prompts for AI image generation tools like DALL·E. The user will provide a transcript of a video, and your task is to:
    
                        1. Analyze the transcript to identify the main themes, key elements, and most visually striking or important topics that could be used to represent the video in a thumbnail.
                        2. Craft a detailed and descriptive prompt that can be used directly in DALL·E to generate a visually appealing thumbnail for the video.
                        3. The prompt should include specific details about the scene, characters, objects, and emotions that need to be depicted, as well as any relevant colors, styles, or themes that would make the thumbnail eye-catching and relevant to the video content.
                        4. Ensure the prompt is structured in a way that DALL·E can easily interpret, with clear instructions on what should be included in the image.
    
                        The final output should be a single, well-crafted prompt that can be used directly in DALL·E to generate the thumbnail.
    
                        The output should be structured as follows:
    
                        ```
                        DALL·E Prompt: "A detailed description of the scene for the thumbnail, including key elements, colors, styles, and emotions."
                        """},
                    {"role": "user", "content": f"Transcript: {transcript}"}
                ]
            )
            # return {"data": completion.choices[0].message}
            dalle_prompt = completion.choices[0].message.content
            dalle_reponse = client.images.generate(
                model="dall-e-3",
                prompt=dalle_prompt,
                n=1,
                size="1024x1024"
            )
            return {"url": dalle_reponse.data[0].url}
        else:
            raise HTTPException(status_code=404, detail=f"File '{item.path}' not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

class SocialImageItem(BaseModel):
    path: str

@app.post("/get_social_image/")
async def get_social_image(item: SocialImageItem):
    try:
        if os.path.exists(item.path):
            with open(item.path, "r") as file:
                transcript = file.read()
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": """
                        You are a creative and insightful language model with expertise in generating prompts for AI image generation tools like DALL·E. The user will provide a transcript of a video, and your task is to:

                        1. Analyze the transcript to identify the main themes, key visuals, and most compelling elements that could be depicted in an image suitable for a social media post.
                        2. Craft a detailed and vivid DALL·E prompt that can be used to create an engaging image for social media. The prompt should capture the essence of the video and translate it into a visually appealing scene.
                        3. Include specific details about the scene, characters, objects, colors, and emotions that should be depicted in the image, as well as any particular style or aesthetic that would make the image stand out on social media platforms.
                        4. Ensure the prompt is structured in a way that DALL·E can easily interpret, with clear and precise instructions on what should be included in the image.

                        The final output should be a single, well-crafted prompt that can be used directly in DALL·E to generate the social media post image.

                        The output should be structured as follows:

                        ```
                        DALL·E Prompt: "A detailed description of the image for the social media post, including key elements, colors, styles, and emotions."
                        """},
                    {"role": "user", "content": f"Transcript: {transcript}"}
                ]
            )
            # return {"data": completion.choices[0].message}
            dalle_prompt = completion.choices[0].message.content
            dalle_reponse = client.images.generate(
                model="dall-e-3",
                prompt=dalle_prompt,
                n=1,
                size="1024x1024"
            )
            return {"url": dalle_reponse.data[0].url}
        else:
            raise HTTPException(status_code=404, detail=f"File '{item.path}' not found.")
    except Exception as e:
        return HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
