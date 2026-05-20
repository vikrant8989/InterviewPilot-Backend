import os
import uuid
import base64
from datetime import datetime, timedelta
from typing import Optional

from gtts import gTTS
from app.core.r2_storage import upload_bytes, mime_to_extension
from app.core.config import settings


async def generate_question_tts_audio_url(
    *,
    session_id: str,
    turn_index: int,
    text: str,
) -> Optional[str]:
    """
    Generate TTS audio for a question.
    Returns a data URL for immediate playback (no R2 upload required).
    """
    print(f"[TTS] Starting audio generation for session {session_id}, turn {turn_index}")
    print(f"[TTS] Text to speak: {text[:100]}...")
    
    try:
        # Generate audio using gTTS
        print(f"[TTS] Creating gTTS object...")
        tts = gTTS(text=text, lang='en', slow=False)
        
        # Save to temporary file
        temp_filename = f"tts_{uuid.uuid4()}.mp3"
        print(f"[TTS] Saving to temp file: {temp_filename}")
        tts.save(temp_filename)
        
        # Read the file
        print(f"[TTS] Reading temp file...")
        with open(temp_filename, 'rb') as f:
            audio_bytes = f.read()
        
        print(f"[TTS] Audio bytes length: {len(audio_bytes)}")
        
        # Clean up temp file
        os.remove(temp_filename)
        print(f"[TTS] Temp file removed")
        
        # Convert to base64 data URL for immediate playback
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        data_url = f"data:audio/mpeg;base64,{audio_base64}"
        
        print(f"[TTS] Data URL generated, length: {len(data_url)}")
        return data_url
        
    except Exception as e:
        print(f"[TTS] Error generating TTS audio: {e}")
        import traceback
        traceback.print_exc()
        return None
