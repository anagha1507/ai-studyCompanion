# Save as debug_youtube.py
from youtube_transcript_api import YouTubeTranscriptApi
import time

video_id = "8S0FDjFBj8o"
print(f"Testing video: {video_id}")

time.sleep(1)

try:
    ytt_api = YouTubeTranscriptApi()
    print("✅ Created API object")
    
    transcript_list = ytt_api.list(video_id)
    print(f"✅ Got transcript list: {list(transcript_list)}")
    
    transcript = transcript_list.find_transcript(['en'])
    print(f"✅ Found English transcript")
    
    fetched = transcript.fetch()
    text = " ".join([chunk.text for chunk in fetched])
    print(f"✅ Text extracted: {len(text)} characters")
    print(f"First 100 chars: {text[:100]}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print(f"Error type: {type(e).__name__}")