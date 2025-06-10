"""
Main application entry point for the Virtual Clinic Voice App
"""

import asyncio
import logging
import sys
from pathlib import Path

from .config import Config
from .voice_client import VoiceClient


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('voice_app.log')
        ]
    )


async def audio_handler(audio_data: bytes):
    """
    Handle incoming audio data from the AI
    In a real application, this would play the audio
    """
    # For now, just log that we received audio
    logging.info(f"Received audio chunk: {len(audio_data)} bytes")
    
    # In a real implementation, you would:
    # 1. Queue the audio data
    # 2. Play it through speakers
    # 3. Handle audio streaming


async def main():
    """Main application function"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Validate configuration
        Config.validate()
        logger.info("Starting Virtual Clinic Voice App...")
        
        # Create voice client with audio handler
        client = VoiceClient(audio_handler=audio_handler)
        
        # Start the conversation
        logger.info("Voice app is ready! The virtual receptionist is standing by.")
        logger.info("Press Ctrl+C to exit.")
        
        await client.start_conversation()
        
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        logger.error("Please check your .env file and ensure OPENAI_API_KEY is set")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)


def run():
    """Entry point for running the application"""
    asyncio.run(main())


if __name__ == "__main__":
    run() 