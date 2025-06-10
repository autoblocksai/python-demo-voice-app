"""
Command Line Interface for the Voice App
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from .main import main as run_main_app
from .config import Config


def setup_cli_logging(verbose: bool = False):
    """Setup logging for CLI"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


async def run_tests():
    """Run the test suite"""
    try:
        from tests.test_voice_app import main as test_main
        await test_main()
    except ImportError:
        print("Error: Could not import test module")
        sys.exit(1)


async def run_audio_file_test(audio_file: str):
    """Run a test with a specific audio file"""
    try:
        from tests.test_voice_app import AudioTestHarness
        
        if not Path(audio_file).exists():
            print(f"Error: Audio file {audio_file} not found")
            sys.exit(1)
        
        harness = AudioTestHarness()
        
        scenarios = [{
            'description': f'Test with audio file: {audio_file}',
            'audio_file': audio_file
        }]
        
        await harness.test_conversation_scenario(scenarios)
        
        # Save conversation
        conversation_file = harness.save_full_conversation(f"test_audio/conversation_with_{Path(audio_file).stem}.wav")
        harness.save_captured_audio("response_to_" + Path(audio_file).stem + ".wav")
        harness.print_conversation_summary()
        
        if conversation_file:
            print(f"\n🎵 Conversation saved to: {conversation_file}")
        
    except Exception as e:
        print(f"Error running audio file test: {e}")
        sys.exit(1)


def show_config():
    """Show current configuration"""
    try:
        config = Config()
        
        print("\nVirtual Clinic Voice App Configuration")
        print("=" * 40)
        print(f"Clinic Name: {config.CLINIC_NAME}")
        print(f"Clinic Hours: {config.CLINIC_HOURS}")
        print(f"Voice Model: {config.VOICE_MODEL}")
        print(f"Voice: {config.VOICE}")
        print(f"Audio Format: {config.AUDIO_FORMAT}")
        print(f"Sample Rate: {config.SAMPLE_RATE}")
        print(f"OpenAI API Key: {'✓ Set' if config.OPENAI_API_KEY else '✗ Not set'}")
        print("=" * 40)
        
    except Exception as e:
        print(f"Error showing configuration: {e}")


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="Virtual Clinic Voice App - A demo receptionist using OpenAI Realtime API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m python_demo_voice_app.cli run                    # Start the voice app
  python -m python_demo_voice_app.cli test                   # Run all tests
  python -m python_demo_voice_app.cli test-audio sample.wav  # Test with audio file
  python -m python_demo_voice_app.cli config                 # Show configuration
        """
    )
    
    parser.add_argument(
        "command",
        choices=["run", "test", "test-audio", "config"],
        help="Command to execute"
    )
    
    parser.add_argument(
        "audio_file",
        nargs="?",
        help="Audio file path (required for test-audio command)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    setup_cli_logging(args.verbose)
    
    if args.command == "run":
        print("Starting Virtual Clinic Voice App...")
        print("Make sure you have set your OPENAI_API_KEY environment variable.")
        print("Press Ctrl+C to exit.\n")
        asyncio.run(run_main_app())
        
    elif args.command == "test":
        print("Running Voice App Tests...")
        asyncio.run(run_tests())
        
    elif args.command == "test-audio":
        if not args.audio_file:
            print("Error: Audio file path is required for test-audio command")
            parser.print_help()
            sys.exit(1)
        
        print(f"Testing with audio file: {args.audio_file}")
        asyncio.run(run_audio_file_test(args.audio_file))
        
    elif args.command == "config":
        show_config()


if __name__ == "__main__":
    main() 