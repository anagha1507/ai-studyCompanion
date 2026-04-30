from youtube_transcript_api import YouTubeTranscriptApi

# Check what methods are available
print("Available methods:", [m for m in dir(YouTubeTranscriptApi) if not m.startswith('_')])

# Test with the correct method for your version
video_id = "8S0FDjFBj8o"

try:
    # Method 1: Using the class instance
    yt = YouTubeTranscriptApi()
    transcript = yt.fetch(video_id)
    print("✅ Method 1 (fetch) works!")
except Exception as e:
    print(f"Method 1 failed: {e}")

try:
    # Method 2: Static method
    transcript = YouTubeTranscriptApi.list_transcripts(video_id)
    print("✅ Method 2 (list_transcripts) works!")
except Exception as e:
    print(f"Method 2 failed: {e}")

try:
    # Method 3: Get transcript directly from list
    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
    transcript = transcript_list.find_transcript(['en']).fetch()
    text = " ".join([entry['text'] for entry in transcript])
    print(f"✅ Method 3 works! Extracted {len(text)} characters")
except Exception as e:
    print(f"Method 3 failed: {e}")