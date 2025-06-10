"""
Voice Client - Handles WebSocket connection to OpenAI Realtime API
"""

import asyncio
import base64
import json
import logging
import websockets
from typing import Callable, Optional, Dict, Any

from .config import Config
from .receptionist import VirtualClinicReceptionist


class VoiceClient:
    """
    Handles the WebSocket connection to OpenAI's Realtime API
    """

    def __init__(self, audio_handler: Optional[Callable] = None):
        self.config = Config()
        self.config.validate()
        
        self.logger = logging.getLogger(__name__)
        self.receptionist = VirtualClinicReceptionist()
        self.websocket = None
        self.audio_handler = audio_handler
        
        # WebSocket URL for OpenAI Realtime API
        self.ws_url = f"wss://api.openai.com/v1/realtime?model={self.config.VOICE_MODEL}"
        
        # Headers for authentication
        self.headers = {
            "Authorization": f"Bearer {self.config.OPENAI_API_KEY}",
            "OpenAI-Beta": "realtime=v1"
        }

    async def connect(self):
        """Establish WebSocket connection"""
        try:
            self.logger.info("Connecting to OpenAI Realtime API...")
            self.websocket = await websockets.connect(
                self.ws_url,
                extra_headers=self.headers
            )
            self.logger.info("Connected successfully!")
            
            # Configure the session
            await self._configure_session()
            
        except Exception as e:
            self.logger.error(f"Failed to connect: {e}")
            raise

    async def _configure_session(self):
        """Configure the session with receptionist settings"""
        config = self.receptionist.get_conversation_config()
        
        session_update = {
            "type": "session.update",
            "session": config
        }
        
        await self._send_message(session_update)
        self.logger.info("🔧 Session configured with receptionist settings")
        self.logger.info(f"🎯 Using voice: {config['voice']}, model: {config['model']}")

    async def _send_message(self, message: Dict[str, Any]):
        """Send a message through the WebSocket"""
        if self.websocket:
            await self.websocket.send(json.dumps(message))
            self.logger.debug(f"Sent message: {message['type']}")

    async def send_audio(self, audio_data: bytes):
        """Send audio data to the API"""
        # Convert audio to base64
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        
        message = {
            "type": "input_audio_buffer.append",
            "audio": audio_base64
        }
        
        await self._send_message(message)

    async def commit_audio(self):
        """Commit the audio buffer to trigger processing"""
        message = {"type": "input_audio_buffer.commit"}
        await self._send_message(message)
        
        # Explicitly trigger response generation
        await self._send_message({"type": "response.create"})

    async def listen(self):
        """Listen for messages from the API"""
        if not self.websocket:
            raise RuntimeError("Not connected to WebSocket")
        
        try:
            async for message in self.websocket:
                await self._handle_message(json.loads(message))
        except websockets.exceptions.ConnectionClosed:
            self.logger.info("WebSocket connection closed")
        except Exception as e:
            self.logger.error(f"Error in listen loop: {e}")
            raise

    async def _handle_message(self, message: Dict[str, Any]):
        """Handle incoming messages from the API"""
        message_type = message.get("type")
        self.logger.info(f"🔄 Received message: {message_type}")
        
        if message_type == "error":
            self.logger.error(f"❌ API Error: {message}")
            
        elif message_type == "session.created":
            self.logger.info("✅ Session created successfully")
            
        elif message_type == "session.updated":
            self.logger.info("✅ Session updated")
            
        elif message_type == "conversation.item.created":
            self._handle_conversation_item(message)
            
        elif message_type == "response.created":
            self.logger.info("🎯 Response creation started")
            
        elif message_type == "response.output_item.added":
            self.logger.info("📝 Response output item added")
            
        elif message_type == "response.audio.delta":
            await self._handle_audio_delta(message)
            
        elif message_type == "response.audio.done":
            self.logger.info("🎵 Audio response completed")
            
        elif message_type == "response.done":
            self.logger.info("✅ Response generation completed")
            
        elif message_type == "response.function_call_arguments.delta":
            self._handle_function_call_delta(message)
            
        elif message_type == "response.function_call_arguments.done":
            await self._handle_function_call_done(message)
            
        elif message_type == "input_audio_buffer.speech_started":
            self.logger.info("🎤 User started speaking")
            
        elif message_type == "input_audio_buffer.speech_stopped":
            self.logger.info("🛑 User stopped speaking")
            
        elif message_type == "input_audio_buffer.committed":
            self.logger.info("✅ Audio buffer committed")
            
        elif message_type == "conversation.item.input_audio_transcription.completed":
            self._handle_transcription(message)
            
        else:
            self.logger.debug(f"🔍 Unhandled message type: {message_type}")

    def _handle_conversation_item(self, message: Dict[str, Any]):
        """Handle conversation item creation"""
        item = message.get("item", {})
        item_type = item.get("type")
        
        if item_type == "message":
            role = item.get("role")
            self.logger.info(f"New {role} message in conversation")

    async def _handle_audio_delta(self, message: Dict[str, Any]):
        """Handle incoming audio chunks"""
        if "delta" in message:
            audio_base64 = message["delta"]
            audio_data = base64.b64decode(audio_base64)
            
            # Pass audio to handler if provided
            if self.audio_handler:
                await self.audio_handler(audio_data)

    def _handle_function_call_delta(self, message: Dict[str, Any]):
        """Handle function call argument deltas"""
        # In a more complex implementation, you might accumulate these deltas
        pass

    async def _handle_function_call_done(self, message: Dict[str, Any]):
        """Handle completed function call"""
        call_id = message.get("call_id")
        name = message.get("name")
        arguments_str = message.get("arguments")
        
        try:
            arguments = json.loads(arguments_str) if arguments_str else {}
            self.logger.info(f"Function call: {name} with args: {arguments}")
            
            # Handle the function call
            result = self.receptionist.handle_function_call(name, arguments)
            
            # Send the result back
            response_message = {
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps(result)
                }
            }
            
            await self._send_message(response_message)
            
            # Trigger response generation
            await self._send_message({"type": "response.create"})
            
        except Exception as e:
            self.logger.error(f"Error handling function call: {e}")

    def _handle_transcription(self, message: Dict[str, Any]):
        """Handle audio transcription"""
        transcript = message.get("transcript", "")
        if transcript:
            self.logger.info(f"User said: {transcript}")

    async def close(self):
        """Close the WebSocket connection"""
        if self.websocket:
            await self.websocket.close()
            self.logger.info("WebSocket connection closed")

    async def start_conversation(self):
        """Start a conversation session"""
        await self.connect()
        
        # Start listening for messages
        listen_task = asyncio.create_task(self.listen())
        
        try:
            await listen_task
        except KeyboardInterrupt:
            self.logger.info("Conversation interrupted by user")
        finally:
            await self.close() 