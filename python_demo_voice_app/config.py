"""
Configuration settings for the voice app
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Application configuration"""
    
    # OpenAI API settings
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_ORG_ID = os.getenv("OPENAI_ORG_ID")
    
    # Voice app settings
    VOICE_MODEL = "gpt-4o-realtime-preview"
    VOICE = "alloy"  # Options: alloy, echo, fable, onyx, nova, shimmer
    
    # Audio settings
    AUDIO_FORMAT = "pcm16"  # PCM 16-bit audio at 24kHz
    SAMPLE_RATE = 24000
    
    # Virtual clinic settings
    CLINIC_NAME = "HealthCare Plus Virtual Clinic"
    CLINIC_HOURS = "Monday-Friday 8AM-6PM, Saturday 9AM-4PM"
    
    @classmethod
    def validate(cls):
        """Validate that required configuration is present"""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        return True 