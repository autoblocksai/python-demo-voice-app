# Virtual Clinic Voice App

A demo voice application that acts as a virtual receptionist for a healthcare clinic, powered by OpenAI's Realtime API.

## Features

- **Real-time Voice Conversation**: Uses OpenAI's Realtime API for natural voice interactions
- **Virtual Clinic Receptionist**: Handles appointment scheduling, doctor availability, and general clinic inquiries
- **Function Calling**: Integrates with backend functions for appointment scheduling and doctor lookups
- **Audio File Testing**: Comprehensive test suite that can process audio files and capture responses
- **Professional Healthcare Context**: Designed specifically for medical receptionist scenarios with appropriate guardrails

## Requirements

- Python 3.12+
- OpenAI API key with Realtime API access
- Poetry for dependency management

## Quick Start

1. **Clone and setup the project**:
   ```bash
   git clone <repository-url>
   cd python-demo-voice-app
   ```

2. **Install dependencies**:
   ```bash
   poetry install
   ```

3. **Set up your environment**:
   ```bash
   cp env.example .env
   # Edit .env and add your OpenAI API key
   ```

4. **Run the voice app**:
   ```bash
   poetry run python -m python_demo_voice_app.cli run
   ```

## Configuration

The app uses environment variables for configuration. Copy `env.example` to `.env` and configure:

```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional
OPENAI_ORG_ID=your_org_id_here
```

You can view the current configuration with:
```bash
poetry run python -m python_demo_voice_app.cli config
```

## Usage

### Running the Voice App

Start the virtual receptionist:
```bash
poetry run python -m python_demo_voice_app.cli run
```

The app will:
1. Connect to OpenAI's Realtime API
2. Configure the session as a virtual clinic receptionist
3. Listen for audio input and respond naturally
4. Handle function calls for appointments and doctor availability

### Testing

Run the complete test suite:
```bash
poetry run python -m python_demo_voice_app.cli test
```

Test with a specific audio file:
```bash
poetry run python -m python_demo_voice_app.cli test-audio path/to/audio.wav
```

The test suite will:
- Send synthetic or real audio to the API
- Capture the AI's audio responses
- Save complete conversations as WAV files
- Generate individual scenario recordings
- Provide detailed conversation summaries

## Virtual Receptionist Capabilities

The AI receptionist can handle:

### Appointment Scheduling
- Check doctor availability
- Schedule appointments with specific doctors
- Collect patient information (name, phone, reason for visit)
- Provide confirmation numbers

### Information Services
- Clinic hours and location information
- Available services and specialties
- Doctor information and specialties
- Insurance and billing questions

### Emergency Triage
- Recognize urgent medical situations
- Direct patients to emergency services when appropriate
- Maintain professional medical boundaries

### Available Doctors
- **Dr. Sarah Smith** - Family Medicine
- **Dr. Michael Johnson** - Internal Medicine
- **Dr. Emily Williams** - Pediatrics

## Project Structure

```
python_demo_voice_app/
├── __init__.py              # Package initialization
├── config.py                # Configuration and settings
├── receptionist.py          # Virtual receptionist logic
├── voice_client.py          # WebSocket client for Realtime API
├── main.py                  # Main application entry point
└── cli.py                   # Command-line interface

tests/
├── __init__.py
└── test_voice_app.py        # Comprehensive test suite

pyproject.toml               # Poetry configuration
env.example                  # Environment variable template
README.md                    # This file
```

## Core Components

### VirtualClinicReceptionist
Handles the conversation logic and function calls:
- System instructions for medical receptionist behavior
- Function definitions for appointments and doctor lookup
- Mock data for doctors and services

### VoiceClient
Manages the WebSocket connection to OpenAI:
- Session configuration
- Audio streaming (send/receive)
- Message handling and function call processing

### AudioTestHarness
Comprehensive testing framework:
- Audio file processing and format conversion
- Conversation scenario testing
- Response capture and analysis

## Audio Processing

The app handles audio in PCM 16-bit format at 24kHz (required by OpenAI Realtime API):

- **Input**: Accepts various audio formats via file upload, converts to required format using pydub
- **Output**: Receives and captures PCM audio chunks from the AI via WebSocket
- **Testing**: Can generate synthetic audio or process real audio files
- **Recording**: Saves complete conversations as WAV files for analysis and demonstration

### Test Audio Files

The app creates comprehensive audio recordings in the `test_audio/` directory:
- **Full conversations** - Complete back-and-forth interactions
- **Individual scenarios** - Separate files for each test case
- **Response analysis** - Captured AI responses for evaluation

## Function Calling

The receptionist can call these functions:

### `schedule_appointment`
Parameters:
- `patient_name`: Patient's full name
- `doctor`: Requested doctor
- `time_slot`: Requested appointment time
- `reason`: Reason for the visit
- `phone`: Patient's contact number

### `check_doctor_availability`
Parameters:
- `specialty`: Optional medical specialty filter

## Development

### Adding New Features

1. **New Functions**: Add to `receptionist.py` in the `get_conversation_config()` method
2. **Function Handlers**: Implement in `receptionist.py` `handle_function_call()` method
3. **Tests**: Add scenarios to `test_voice_app.py`

### Logging

The app uses Python's logging module:
- **INFO**: General application flow
- **DEBUG**: Detailed WebSocket and API interactions
- **ERROR**: Error conditions and failures

Logs are written to both console and `voice_app.log`.

## Troubleshooting

### Common Issues

1. **Missing OpenAI API Key**
   ```
   Error: OPENAI_API_KEY environment variable is required
   ```
   Solution: Set your API key in the `.env` file

2. **Connection Issues**
   ```
   Error: Failed to connect to WebSocket
   ```
   Solution: Check your internet connection and API key validity

3. **Audio Format Issues**
   ```
   Error loading audio file
   ```
   Solution: Ensure audio files are in supported formats (WAV, MP3, etc.)

### Debug Mode

Run with verbose logging:
```bash
poetry run python -m python_demo_voice_app.cli run -v
```

## Security Considerations

- **API Keys**: Never commit API keys to version control
- **Medical Information**: This is a demo app - don't use for real patient data
- **Function Calls**: Validate all function inputs in production
- **Audio Storage**: Be mindful of audio data privacy

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is for demonstration purposes. Please ensure compliance with healthcare regulations if adapting for real medical use.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the logs for error details
3. Ensure your OpenAI API key has Realtime API access
4. Verify all dependencies are installed correctly
