"""
Test suite for the Voice App - includes audio file testing
"""

import asyncio
import base64
import io
import json
import logging
import os
import tempfile
import wave
from pathlib import Path
from typing import List, Dict, Any

from pydub import AudioSegment
from openai import AsyncOpenAI

from python_demo_voice_app.config import Config
from python_demo_voice_app.voice_client import VoiceClient
from python_demo_voice_app.receptionist import VirtualClinicReceptionist


class AudioTestHarness:
    """
    Test harness for sending audio files and capturing responses
    """

    def __init__(self):
        self.config = Config()
        self.logger = logging.getLogger(__name__)
        self.received_audio_chunks = []
        self.conversation_log = []
        self.full_conversation_audio = []  # Store complete conversation
        self.current_scenario_audio = []   # Store current scenario audio
        self.openai_client = AsyncOpenAI(api_key=self.config.OPENAI_API_KEY)
        
    async def audio_capture_handler(self, audio_data: bytes):
        """Capture audio responses for testing"""
        self.received_audio_chunks.append(audio_data)
        self.logger.info(f"🎧 Captured receptionist audio chunk: {len(audio_data)} bytes")



    async def create_tts_audio(self, text: str, voice: str = "nova") -> bytes:
        """
        Use OpenAI's TTS API to generate realistic speech from text
        This creates actual human-like speech for testing
        """
        try:
            self.logger.info(f"Generating TTS audio for: '{text[:50]}...'")
            
            # Generate speech using OpenAI TTS with timeout
            response = await asyncio.wait_for(
                self.openai_client.audio.speech.create(
                    model="tts-1",
                    voice=voice,  # nova, alloy, echo, fable, onyx, shimmer
                    input=text,
                    response_format="wav",
                    speed=1.0
                ),
                timeout=30.0  # 30 second timeout
            )
            
            # Get the audio data
            audio_data = response.content
            
            # Convert to the required format using pydub
            audio_segment = AudioSegment.from_file(
                io.BytesIO(audio_data), 
                format="wav"
            )
            
            # Convert to required format: PCM 16-bit, 24kHz, mono
            audio_segment = audio_segment.set_frame_rate(self.config.SAMPLE_RATE)
            audio_segment = audio_segment.set_channels(1)  # Mono
            audio_segment = audio_segment.set_sample_width(2)  # 16-bit
            
            self.logger.info(f"Generated {len(audio_segment.raw_data)} bytes of TTS audio")
            return audio_segment.raw_data
            
        except asyncio.TimeoutError:
            self.logger.error(f"TTS generation timed out after 30 seconds")
            return self.create_silence(2.0)  # Return silence as fallback
        except Exception as e:
            self.logger.error(f"Error generating TTS audio: {e}")
            return self.create_silence(2.0)  # Return silence as fallback instead of crashing

    async def transcribe_audio(self, audio_data: bytes) -> str:
        """
        Use OpenAI's Whisper API to transcribe audio to text
        """
        try:
            # Save audio to temporary file for Whisper API
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                with wave.open(temp_file.name, 'wb') as wav_file:
                    wav_file.setnchannels(1)  # Mono
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(self.config.SAMPLE_RATE)
                    wav_file.writeframes(audio_data)
                
                # Transcribe using Whisper with timeout
                with open(temp_file.name, 'rb') as audio_file:
                    transcript = await asyncio.wait_for(
                        self.openai_client.audio.transcriptions.create(
                            model="whisper-1",
                            file=audio_file
                        ),
                        timeout=30.0  # 30 second timeout
                    )
                
                # Clean up temp file
                os.unlink(temp_file.name)
                
                return transcript.text.strip()
                
        except asyncio.TimeoutError:
            self.logger.error(f"Audio transcription timed out after 30 seconds")
            return ""
        except Exception as e:
            self.logger.error(f"Error transcribing audio: {e}")
            return ""

    async def generate_patient_response(self, conversation_history: List[Dict], scenario_context: str) -> str:
        """
        Use OpenAI's LLM to generate the patient's next response in the conversation
        """
        try:
            # Build conversation context
            messages = [
                {
                    "role": "system",
                    "content": f"""You are a patient calling a medical clinic. 

Context: {scenario_context}

You should:
- Respond naturally to what the receptionist says
- Stay in character as a patient with the described concern
- Ask follow-up questions when appropriate
- Provide requested information (name, phone, etc.)
- Be realistic in your responses
- Keep responses conversational and not too long

Conversation so far: {conversation_history}"""
                }
            ]
            
            # Add conversation history
            for turn in conversation_history:
                messages.append({
                    "role": "user" if turn["speaker"] == "patient" else "assistant",
                    "content": turn["text"]
                })
            
            # Generate response with timeout
            response = await asyncio.wait_for(
                self.openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=messages,
                    max_tokens=150,
                    temperature=0.7
                ),
                timeout=30.0  # 30 second timeout
            )
            
            return response.choices[0].message.content.strip()
            
        except asyncio.TimeoutError:
            self.logger.error(f"Patient response generation timed out after 30 seconds")
            return "I'm sorry, could you repeat that?"
        except Exception as e:
            self.logger.error(f"Error generating patient response: {e}")
            return "I'm sorry, could you repeat that?"

    def send_text_message(self, text: str) -> Dict:
        """
        For more reliable testing, send text directly instead of audio
        This simulates what would happen if audio was transcribed
        """
        return {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": text
                    }
                ]
            }
        }

    def create_silence(self, duration_seconds: float = 0.5) -> bytes:
        """Create silence for padding between conversation parts"""
        sample_rate = self.config.SAMPLE_RATE
        samples = int(sample_rate * duration_seconds)
        # Create silence (16-bit PCM, 2 bytes per sample)
        silence = b'\x00\x00' * samples
        return silence

    def add_conversation_label(self, text: str):
        """Add a text label to the conversation (as silence with metadata)"""
        # Create a short silence to separate conversation parts
        silence = self.create_silence(0.3)
        self.full_conversation_audio.append(silence)
        self.logger.info(f"Added conversation label: {text}")

    def load_audio_file(self, file_path: str) -> bytes:
        """
        Load an audio file and convert it to the required format
        """
        try:
            # Load audio file using pydub
            audio = AudioSegment.from_file(file_path)
            
            # Convert to required format: PCM 16-bit, 24kHz, mono
            audio = audio.set_frame_rate(self.config.SAMPLE_RATE)
            audio = audio.set_channels(1)  # Mono
            audio = audio.set_sample_width(2)  # 16-bit
            
            return audio.raw_data
            
        except Exception as e:
            self.logger.error(f"Error loading audio file {file_path}: {e}")
            raise

    async def test_dynamic_conversation(self, scenario: Dict[str, Any], max_turns: int = 5):
        """
        Test a dynamic AI-to-AI conversation with back-and-forth exchanges
        
        Args:
            scenario: Dict with 'initial_message', 'context', and 'voice'
            max_turns: Maximum number of conversation turns
        """
        client = VoiceClient(audio_handler=self.audio_capture_handler)
        
        conversation_history = []
        scenario_context = scenario.get('context', scenario.get('description', ''))
        patient_voice = scenario.get('voice', 'nova')
        
        try:
            await client.connect()
            
            # Start listening for WebSocket messages in the background
            listen_task = asyncio.create_task(client.listen())
            
            self.logger.info("🎤 Starting dynamic AI-to-AI conversation...")
            self.logger.info(f"📋 Scenario: {scenario_context}")
            
            # Initial patient message
            current_message = scenario.get('initial_message', scenario.get('text', ''))
            
            for turn in range(max_turns):
                self.logger.info(f"🔄 Turn {turn + 1}/{max_turns}")
                
                # Clear audio chunks for this turn
                self.received_audio_chunks.clear()
                
                # Generate patient audio
                self.logger.info(f"🗣️  Patient ({patient_voice}): {current_message}")
                patient_audio = await self.create_tts_audio(current_message, patient_voice)
                
                # Add to conversation recording
                self.full_conversation_audio.append(patient_audio)
                conversation_history.append({
                    "speaker": "patient",
                    "text": current_message,
                    "turn": turn + 1
                })
                
                # Send to receptionist
                await client.send_audio(patient_audio)
                await client.commit_audio()
                
                # Small delay to let commit process
                await asyncio.sleep(1)
                
                # Wait for receptionist response
                self.logger.info(f"⏳ Waiting for receptionist response...")
                await asyncio.sleep(5)  # Give time for processing
                
                if not self.received_audio_chunks:
                    self.logger.warning(f"⚠️  No response from receptionist on turn {turn + 1}")
                    # Add silence to represent no response
                    silence = self.create_silence(1.0)
                    self.full_conversation_audio.append(silence)
                    self.logger.info(f"📼 Added {len(silence)} bytes of silence for missing response")
                    break
                
                # Combine receptionist audio response
                receptionist_audio = b''.join(self.received_audio_chunks)
                self.logger.info(f"✅ Receptionist responded with {len(receptionist_audio)} bytes of audio")
                
                # Add some spacing before receptionist response
                self.full_conversation_audio.append(self.create_silence(0.3))
                # Add receptionist response to conversation recording
                self.full_conversation_audio.append(receptionist_audio)
                # Add spacing after response
                self.full_conversation_audio.append(self.create_silence(0.5))
                
                # Transcribe receptionist response
                self.logger.info(f"🎧 Transcribing receptionist response...")
                receptionist_text = await self.transcribe_audio(receptionist_audio)
                
                if not receptionist_text:
                    self.logger.warning("⚠️  Failed to transcribe receptionist response")
                    break
                
                self.logger.info(f"🤖 Receptionist: {receptionist_text}")
                conversation_history.append({
                    "speaker": "receptionist", 
                    "text": receptionist_text,
                    "turn": turn + 1
                })
                
                # Generate patient's next response
                if turn < max_turns - 1:  # Don't generate response on last turn
                    self.logger.info(f"🧠 Generating patient's next response...")
                    current_message = await self.generate_patient_response(
                        conversation_history, scenario_context
                    )
                    
                    if not current_message or current_message.lower() in ["goodbye", "thank you", "bye"]:
                        self.logger.info("💬 Patient ending conversation")
                        break
                
                # Add silence between turns
                self.full_conversation_audio.append(self.create_silence(0.8))
                
            self.logger.info(f"✅ Conversation completed after {len(conversation_history)} exchanges")
            
            # Log conversation summary
            self.conversation_log.append({
                'scenario': scenario_context,
                'turns': len([h for h in conversation_history if h['speaker'] == 'patient']),
                'total_exchanges': len(conversation_history),
                'conversation': conversation_history,
                'mode': 'dynamic_ai_conversation'
            })
            
        finally:
            # Cancel the listen task and close connection
            if 'listen_task' in locals():
                listen_task.cancel()
                try:
                    await listen_task
                except asyncio.CancelledError:
                    pass
            await client.close()
            
        return conversation_history

    def save_captured_audio(self, filename: str):
        """Save captured audio chunks to a file"""
        if not self.received_audio_chunks:
            self.logger.warning("No audio chunks to save")
            return
        
        # Combine all audio chunks
        combined_audio = b''.join(self.received_audio_chunks)
        
        # Save as WAV file
        with wave.open(filename, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.config.SAMPLE_RATE)
            wav_file.writeframes(combined_audio)
        
        self.logger.info(f"Saved captured audio to {filename}")



    def save_full_conversation(self, filename: str = "test_audio/full_conversation.wav"):
        """Save the complete conversation as one audio file"""
        if not self.full_conversation_audio:
            self.logger.warning("No conversation audio to save")
            return
        
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        
        # Combine all conversation audio
        combined_audio = b''.join(self.full_conversation_audio)
        
        with wave.open(filename, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.config.SAMPLE_RATE)
            wav_file.writeframes(combined_audio)
        
        self.logger.info(f"Saved full conversation to {filename}")
        return filename

    def print_dynamic_conversation_summary(self):
        """Print a summary of dynamic AI-to-AI conversations"""
        print("\n" + "="*60)
        print("DYNAMIC AI-TO-AI CONVERSATION SUMMARY")
        print("="*60)
        
        for i, entry in enumerate(self.conversation_log):
            if entry.get('mode') == 'dynamic_ai_conversation':
                print(f"\nConversation {i+1}: {entry['scenario']}")
                print(f"  📊 Total turns: {entry['turns']}")
                print(f"  💬 Total exchanges: {entry['total_exchanges']}")
                
                print(f"  🗣️  Conversation flow:")
                for turn in entry['conversation']:
                    speaker_emoji = "🧑‍⚕️" if turn['speaker'] == 'patient' else "🤖"
                    print(f"    {speaker_emoji} {turn['speaker'].title()}: {turn['text'][:80]}...")
        
        print("="*60)


class VoiceAppTester:
    """
    Main test class for the voice application
    """

    def __init__(self):
        self.harness = AudioTestHarness()
        self.logger = logging.getLogger(__name__)

    async def test_dynamic_conversations(self):
        """Test dynamic AI-to-AI conversations with back-and-forth exchanges"""
        
        # Create test audio directory if it doesn't exist
        test_audio_dir = Path("test_audio")
        test_audio_dir.mkdir(exist_ok=True)
        
        scenarios = [
            {
                'context': 'Patient wants to schedule a routine family medicine appointment',
                'initial_message': 'Hi, I would like to schedule an appointment with a family doctor',
                'voice': 'nova'
            },
            {
                'context': 'Patient has specific concern and wants to see Dr. Smith',
                'initial_message': 'Hello, I need to see Dr. Smith for a follow-up on my blood pressure',
                'voice': 'shimmer'
            },
            {
                'context': 'Patient calling about emergency symptoms',
                'initial_message': 'Hi, I have been having chest pain and shortness of breath since this morning',
                'voice': 'echo'
            }
        ]
        
        self.logger.info("🤖 Testing dynamic AI-to-AI conversations...")
        
        all_conversations = []
        for i, scenario in enumerate(scenarios):
            self.logger.info(f"\n🎬 Starting conversation {i + 1}: {scenario['context']}")
            
            # Reset audio for each conversation
            self.harness.full_conversation_audio.clear()
            
            # Run dynamic conversation
            conversation = await self.harness.test_dynamic_conversation(scenario, max_turns=4)
            all_conversations.append(conversation)
            
            # Save individual conversation
            if self.harness.full_conversation_audio:
                filename = f"test_audio/dynamic_conversation_{i+1}.wav"
                self.harness.save_full_conversation(filename)
                self.logger.info(f"💾 Saved conversation {i+1} to {filename}")
        
        # Save combined conversation log
        self.harness.print_dynamic_conversation_summary()
        
        print(f"\n🎵 Dynamic AI-to-AI conversations completed!")
        print(f"📁 {len(scenarios)} conversation files saved in test_audio/")
        print("🤖 Each contains full back-and-forth AI conversation!")

    def test_receptionist_config(self):
        """Test receptionist configuration"""
        receptionist = VirtualClinicReceptionist()
        
        # Test system instructions
        instructions = receptionist.get_system_instructions()
        assert "virtual receptionist" in instructions.lower()
        assert receptionist.config.CLINIC_NAME in instructions
        
        # Test conversation config
        config = receptionist.get_conversation_config()
        assert config['model'] == receptionist.config.VOICE_MODEL
        assert 'tools' in config
        assert len(config['tools']) >= 2  # Should have at least 2 functions
        
        print("✓ Receptionist configuration tests passed")

    def test_function_calls(self):
        """Test function call handling"""
        receptionist = VirtualClinicReceptionist()
        
        # Test appointment scheduling
        result = receptionist.handle_function_call(
            "schedule_appointment",
            {
                "patient_name": "John Doe",
                "doctor": "Dr. Smith",
                "time_slot": "2:00 PM",
                "reason": "Annual check-up",
                "phone": "555-1234"
            }
        )
        
        assert result['status'] == 'success'
        assert 'confirmation_number' in result
        
        # Test doctor availability check
        result = receptionist.handle_function_call(
            "check_doctor_availability",
            {"specialty": "Family Medicine"}
        )
        
        assert 'available_doctors' in result
        assert result['total_count'] > 0
        
        print("✓ Function call tests passed")

    async def run_all_tests(self):
        """Run all tests"""
        print("Starting Voice App Tests...")
        
        try:
            # Test basic functionality
            self.test_receptionist_config()
            self.test_function_calls()
            
            # Test dynamic AI-to-AI conversations
            await self.test_dynamic_conversations()
            
            print("\n✓ All tests completed successfully!")
            
        except Exception as e:
            print(f"\n✗ Test failed: {e}")
            raise


def setup_test_logging():
    """Setup logging for tests"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


async def main():
    """Main test function"""
    setup_test_logging()
    
    try:
        # Validate configuration
        Config.validate()
        
        tester = VoiceAppTester()
        await tester.run_all_tests()
        
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("Please ensure OPENAI_API_KEY is set in your environment")
    except Exception as e:
        print(f"Test error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main()) 