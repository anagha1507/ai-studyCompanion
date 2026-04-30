from youtube_transcript_api import YouTubeTranscriptApi

# Test video ID
video_id = "8S0FDjFBj8o"

print(f"Testing video ID: {video_id}")

try:
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    text = " ".join([entry['text'] for entry in transcript])
    print(f"✅ SUCCESS! Extracted {len(text)} characters")
    print(f"First 200 chars: {text[:200]}...")
except Exception as e:
    print(f"❌ FAILED: {e}")