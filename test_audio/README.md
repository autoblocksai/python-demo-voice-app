# Test Audio Directory

This directory contains audio recordings generated during testing of the Virtual Clinic Voice App.

## Generated Files

When you run tests, the following files will be created:

### Complete Conversations
- **`full_conversation.wav`** - The entire test conversation from start to finish
- **`conversation_with_[filename].wav`** - Complete conversation when testing with a specific audio file

### Individual Scenarios
- **`scenario_1_greeting_and_appointment_request.wav`** - First test scenario
- **`scenario_2_specific_doctor_request.wav`** - Second test scenario
- **`scenario_3_ask_about_clinic_hours.wav`** - Third test scenario
- **`scenario_4_emergency_triage.wav`** - Fourth test scenario

### Response Only (Legacy)
- **`last_response.wav`** - Just the AI's response from the last scenario
- **`response_to_[filename].wav`** - AI response to a specific input file

## Audio Format

All files are saved as:
- **Format**: WAV (uncompressed)
- **Sample Rate**: 24kHz
- **Channels**: Mono
- **Bit Depth**: 16-bit PCM

## File Structure

Each scenario file contains:
1. **Input Audio** - Your synthetic or uploaded audio
2. **Brief Silence** - 0.5 second pause
3. **AI Response** - The virtual receptionist's reply

The full conversation file contains all scenarios in sequence with appropriate pauses and labels.

## Testing Your Own Audio

Place your audio files anywhere and test them with:
```bash
poetry run voice-app test-audio path/to/your/audio.wav
```

The system will automatically convert your audio to the correct format and generate conversation recordings.

## Listening to Results

Use any audio player to listen to the generated files:
- **macOS**: `open full_conversation.wav`
- **Windows**: `start full_conversation.wav`
- **Linux**: `xdg-open full_conversation.wav`

Or use audio editing software like Audacity for detailed analysis.
