"""
Autoblocks-integrated test suite for the Voice App
Uses Autoblocks scenarios to drive patient conversations
"""

import asyncio
import io
import logging
import os
import tempfile
import uuid
import wave
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import boto3
from autoblocks.api.app_client import AutoblocksAppClient
from autoblocks.scenarios.models import Message
from autoblocks.scenarios.utils import get_selected_scenario_ids
from autoblocks.testing.models import BaseTestCase
from autoblocks.testing.v2.run import run_test_suite
from autoblocks.tracer import init_auto_tracer
from botocore.exceptions import NoCredentialsError, ClientError
from dotenv import load_dotenv
from openai import AsyncOpenAI
from opentelemetry.instrumentation.openai import OpenAIInstrumentor
from pydub import AudioSegment

from python_demo_voice_app.config import Config
from python_demo_voice_app.voice_client import VoiceClient
from .voice_evaluators import (
    VoiceNaturalness,
    MedicalProfessionalism,
    AppointmentHandling,
    PatientEmpathy,
    InformationGathering,
    ConversationFlow,
    UrgencyAssessment,
    CallResolution,
)

# Load environment variables and setup tracing
load_dotenv()
init_auto_tracer(api_key=os.getenv("AUTOBLOCKS_V2_API_KEY"), is_batch_disabled=True)
OpenAIInstrumentor().instrument()

# Initialize Autoblocks client
client = AutoblocksAppClient(
    app_slug="virtual-clinic-receptionist",
)

max_turns = 8


@dataclass
class VoiceConversationOutput:
    """Output from a voice conversation test"""
    messages: List[Dict[str, Any]]
    recording_url: str
    total_turns: int
    conversation_duration_seconds: float
    patient_voice: str
    receptionist_responses_count: int


@dataclass
class VoiceTestCase(BaseTestCase):
    """Test case for voice conversations"""
    scenario_id: str
    patient_voice: str = "nova"

    def hash(self) -> str:
        return f"{self.scenario_id}_{self.patient_voice}"


class AutoblocksVoiceTestHarness:
    """
    Test harness that combines Autoblocks scenario management 
    with voice conversation testing
    """

    def __init__(self):
        self.config = Config()
        self.config.validate()
        self.openai_client = AsyncOpenAI(api_key=self.config.OPENAI_API_KEY)
        self.logger = logging.getLogger(__name__)
        
        # Audio storage
        self.received_audio_chunks = []
        self.full_conversation_audio = []
        
        # Setup R2 client
        self.r2_bucket = "autoblocks-storage"
        self.r2_client = self._setup_r2_client()
        
        # Create test audio directory (as fallback)
        self.test_audio_dir = Path("test_audio/autoblocks")
        self.test_audio_dir.mkdir(parents=True, exist_ok=True)

    def _setup_r2_client(self):
        """Setup R2 (S3-compatible) client"""
        try:
            # R2 credentials should be set via environment variables:
            # R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_ENDPOINT_URL
            return boto3.client(
                's3',
                endpoint_url=os.getenv('R2_ENDPOINT_URL'),
                aws_access_key_id=os.getenv('R2_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('R2_SECRET_ACCESS_KEY'),
                region_name='auto'  # R2 uses 'auto' as region
            )
        except Exception as e:
            self.logger.warning(f"Failed to setup R2 client: {e}")
            return None

    async def audio_capture_handler(self, audio_data: bytes):
        """Capture audio responses from the receptionist"""
        self.received_audio_chunks.append(audio_data)
        self.logger.info(f"🎧 Captured receptionist audio chunk: {len(audio_data)} bytes")

    async def create_tts_audio(self, text: str, voice: str = "nova") -> bytes:
        """Generate TTS audio from text using OpenAI"""
        try:
            self.logger.info(f"🗣️ Generating TTS ({voice}): '{text[:50]}...'")
            
            response = await asyncio.wait_for(
                self.openai_client.audio.speech.create(
                    model="gpt-4o-mini-tts",
                    voice=voice,
                    input=text,
                    response_format="wav",
                    speed=1.0
                ),
                timeout=30.0
            )
            
            # Convert to required format
            audio_data = response.content
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_data), format="wav")
            audio_segment = audio_segment.set_frame_rate(self.config.SAMPLE_RATE)
            audio_segment = audio_segment.set_channels(1)
            audio_segment = audio_segment.set_sample_width(2)
            
            self.logger.info(f"✅ Generated {len(audio_segment.raw_data)} bytes of TTS audio")
            return audio_segment.raw_data
            
        except asyncio.TimeoutError:
            self.logger.error("TTS generation timed out")
            return self.create_silence(2.0)
        except Exception as e:
            self.logger.error(f"TTS generation failed: {e}")
            return self.create_silence(2.0)

    async def transcribe_audio(self, audio_data: bytes) -> str:
        """Transcribe audio to text using Whisper"""
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                with wave.open(temp_file.name, 'wb') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(self.config.SAMPLE_RATE)
                    wav_file.writeframes(audio_data)
                
                with open(temp_file.name, 'rb') as audio_file:
                    transcript = await asyncio.wait_for(
                        self.openai_client.audio.transcriptions.create(
                            model="gpt-4o-mini-transcribe",
                            file=audio_file
                        ),
                        timeout=30.0
                    )
                
                os.unlink(temp_file.name)
                return transcript.text.strip()
                
        except asyncio.TimeoutError:
            self.logger.error("Audio transcription timed out")
            return ""
        except Exception as e:
            self.logger.error(f"Transcription failed: {e}")
            return ""

    def create_silence(self, duration_seconds: float = 0.5) -> bytes:
        """Create silence padding"""
        sample_rate = self.config.SAMPLE_RATE
        samples = int(sample_rate * duration_seconds)
        return b'\x00\x00' * samples

    async def run_voice_conversation(self, test_case: VoiceTestCase) -> VoiceConversationOutput:
        """
        Run a voice conversation using Autoblocks scenario management
        """
        import time
        start_time = time.time()
        
        voice_client = VoiceClient(audio_handler=self.audio_capture_handler)
        conversation_messages = []
        receptionist_responses = 0
        
        try:
            # Connect and start listening
            await voice_client.connect()
            listen_task = asyncio.create_task(voice_client.listen())
            
            self.logger.info(f"🎭 Starting Autoblocks scenario: {test_case.scenario_id}")
            self.logger.info(f"🎤 Patient voice: {test_case.patient_voice}")
            
            # Reset audio storage
            self.full_conversation_audio.clear()
            
            turn = 1
            while turn <= max_turns:
                self.logger.info(f"🔄 Turn {turn}/{max_turns}")
                
                # Get next message from Autoblocks scenario
                all_messages = [
                    Message(role=msg["role"], content=msg["content"]) 
                    for msg in conversation_messages
                ]
                
                try:
                    next_message = client.scenarios.generate_message(
                        scenario_id=test_case.scenario_id, 
                        messages=all_messages
                    )
                    
                    patient_text = next_message.message
                    self.logger.info(f"👤 Patient: {patient_text}")
                    
                    # Add patient message to conversation
                    entry = {
                        "id": str(uuid.uuid4()),
                        "timestamp": datetime.now().replace(microsecond=0).isoformat() + "Z",
                        "role": "user",
                        "content": patient_text,
                    }
                    conversation_messages.append(entry)
                    
                    # Generate patient TTS audio
                    patient_audio = await self.create_tts_audio(patient_text, test_case.patient_voice)
                    self.full_conversation_audio.append(patient_audio)
                    
                    # Clear receptionist audio buffer
                    self.received_audio_chunks.clear()
                    
                    # Send to receptionist
                    await voice_client.send_audio(patient_audio)
                    await voice_client.commit_audio()
                    
                    # Wait for receptionist response
                    await asyncio.sleep(1)  # Brief pause
                    self.logger.info("⏳ Waiting for receptionist response...")
                    await asyncio.sleep(6)  # Wait for processing
                    
                    if self.received_audio_chunks:
                        # Capture receptionist response
                        receptionist_audio = b''.join(self.received_audio_chunks)
                        receptionist_responses += 1
                        
                        self.logger.info(f"✅ Receptionist responded with {len(receptionist_audio)} bytes")
                        
                        # Add spacing and receptionist audio
                        self.full_conversation_audio.append(self.create_silence(0.3))
                        self.full_conversation_audio.append(receptionist_audio)
                        self.full_conversation_audio.append(self.create_silence(0.5))
                        
                        # Transcribe receptionist response
                        receptionist_text = await self.transcribe_audio(receptionist_audio)
                        
                        if receptionist_text:
                            self.logger.info(f"🤖 Receptionist: {receptionist_text}")
                            entry = {
                                "id": str(uuid.uuid4()),
                                "timestamp": datetime.now().replace(microsecond=0).isoformat() + "Z",
                                "role": "assistant",
                                "content": receptionist_text,
                            }
                            conversation_messages.append(entry)
                        else:
                            self.logger.warning("Failed to transcribe receptionist response")
                    else:
                        self.logger.warning("No receptionist response received")
                        # Add silence for missing response
                        self.full_conversation_audio.append(self.create_silence(1.0))
                    
                    # Check if this was the final message
                    if next_message.is_final_message:
                        self.logger.info("🏁 Scenario marked as complete")
                        break
                        
                except Exception as e:
                    self.logger.error(f"Error generating next message: {e}")
                    break
                
                turn += 1
            
            # Save conversation audio
            duration = time.time() - start_time
            audio_filename = f"conversation_{test_case.scenario_id}_{test_case.patient_voice}.wav"
            audio_path = self.save_conversation_audio(audio_filename)
            
            return VoiceConversationOutput(
                messages=conversation_messages,
                recording_url=audio_path,
                total_turns=turn - 1,
                conversation_duration_seconds=duration,
                patient_voice=test_case.patient_voice,
                receptionist_responses_count=receptionist_responses
            )
            
        finally:
            # Cleanup
            if 'listen_task' in locals():
                listen_task.cancel()
                try:
                    await listen_task
                except asyncio.CancelledError:
                    pass
            await voice_client.close()

    def save_conversation_audio(self, filename: str) -> str:
        """Save the full conversation audio to R2 bucket"""
        if not self.full_conversation_audio:
            self.logger.warning("No conversation audio to save")
            return None
        
        combined_audio = b''.join(self.full_conversation_audio)
        
        # Create audio data in memory
        audio_buffer = io.BytesIO()
        with wave.open(audio_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.config.SAMPLE_RATE)
            wav_file.writeframes(combined_audio)
        
        audio_buffer.seek(0)
        audio_data = audio_buffer.getvalue()
        
        # Generate R2 path: python-voice-demo-app/date/some-id/file.wav
        current_date = datetime.now().strftime("%Y-%m-%d")
        unique_id = str(uuid.uuid4())
        r2_key = f"python-voice-demo-app/{current_date}/{unique_id}/{filename}"
        
        # Try to upload to R2, fallback to local storage
        if self.r2_client:
            try:
                self.r2_client.put_object(
                    Bucket=self.r2_bucket,
                    Key=r2_key,
                    Body=audio_data,
                    ContentType='audio/wav'
                )
                # Generate Autoblocks storage URL
                storage_url = f"http://storage.autoblocks.ai/{r2_key}"
                self.logger.info(f"☁️ Uploaded conversation audio to R2: {storage_url}")
                return storage_url
                
            except (NoCredentialsError, ClientError) as e:
                self.logger.error(f"Failed to upload to R2: {e}")
                self.logger.info("Falling back to local storage")
        
        # Fallback to local storage
        file_path = self.test_audio_dir / filename
        with open(file_path, 'wb') as f:
            f.write(audio_data)
        
        self.logger.info(f"💾 Saved conversation audio locally: {file_path}")
        return str(file_path)


def run_autoblocks_voice_tests():
    """
    Main function to run Autoblocks-integrated voice tests
    """
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    try:
        # Load scenarios from Autoblocks
        scenarios = client.scenarios.list_scenarios()
        selected_scenario_ids = get_selected_scenario_ids()
        
        # Filter scenarios if specific ones are selected
        if selected_scenario_ids:
            scenarios = [scenario for scenario in scenarios if scenario.id in selected_scenario_ids]
            logger.info(f"Running {len(scenarios)} selected scenarios")
        else:
            logger.info(f"Running all {len(scenarios)} scenarios")
        
        # Create test cases with different voices
        voices = ["nova"]
        test_cases = []
        
        for scenario in scenarios:
            for voice in voices:
                test_cases.append(VoiceTestCase(
                    scenario_id=scenario.id,
                    patient_voice=voice
                ))
        
        logger.info(f"Created {len(test_cases)} test cases")
        
        # Create test harness
        harness = AutoblocksVoiceTestHarness()
        
        async def test_fn(test_case: VoiceTestCase) -> VoiceConversationOutput:
            """Test function that runs a voice conversation"""
            try:
                return await harness.run_voice_conversation(test_case)
            except Exception as e:
                logger.error(f"Test failed for {test_case.scenario_id}: {e}")
                raise
        
        # Run the test suite
        run_test_suite(
            id="virtual-clinic-receptionist",
            app_slug="virtual-clinic-receptionist",
            test_cases=test_cases,
            fn=test_fn,
            evaluators=[
                VoiceNaturalness(),
                MedicalProfessionalism(),
                AppointmentHandling(),
                PatientEmpathy(),
                InformationGathering(),
                ConversationFlow(),
                UrgencyAssessment(),
                CallResolution(),
            ],
        )
        
        logger.info("🎉 Autoblocks voice test suite completed!")
        logger.info(f"📁 Audio files saved in: {harness.test_audio_dir}")
        
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        raise


if __name__ == "__main__":
    run_autoblocks_voice_tests() 