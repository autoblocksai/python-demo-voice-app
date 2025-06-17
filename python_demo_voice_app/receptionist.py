"""
Virtual Clinic Receptionist - Core conversation logic
"""

import logging
from typing import Dict
from typing import Optional

from .config import Config


class VirtualClinicReceptionist:
    """
    A virtual receptionist for a healthcare clinic that can handle:
    - Appointment scheduling
    - General inquiries about services
    - Doctor availability
    - Insurance questions
    - Emergency triage
    """

    def __init__(self):
        self.config = Config()
        self.logger = logging.getLogger(__name__)

        # Mock data - in a real app, this would come from a database
        self.doctors = {
            "dr_smith": {
                "name": "Dr. Sarah Smith",
                "specialty": "Family Medicine",
                "available_slots": ["9:00 AM", "2:00 PM", "4:30 PM"],
            },
            "dr_johnson": {
                "name": "Dr. Michael Johnson",
                "specialty": "Internal Medicine",
                "available_slots": ["10:30 AM", "1:00 PM", "3:30 PM"],
            },
            "dr_williams": {
                "name": "Dr. Emily Williams",
                "specialty": "Pediatrics",
                "available_slots": ["8:30 AM", "11:00 AM", "2:30 PM"],
            },
        }

        self.services = [
            "Annual check-ups",
            "Vaccinations",
            "Blood tests",
            "X-rays",
            "Physical therapy consultations",
            "Prescription refills",
            "Specialist referrals",
        ]

    def get_system_instructions(self) -> str:
        """Get the system instructions for the AI receptionist"""
        return f"""
You are a friendly and professional virtual receptionist for {self.config.CLINIC_NAME}.

Your responsibilities include:
1. Greeting patients warmly and professionally
2. Helping schedule appointments with available doctors
3. Providing information about clinic services and hours
4. Answering questions about insurance and billing
5. Handling prescription refill requests
6. Triaging urgent medical concerns (directing to emergency services if needed)
7. Collecting basic patient information for appointments

IMPORTANT GUIDELINES:
- Always maintain a warm, professional, and empathetic tone
- Never provide medical advice or diagnose conditions
- For urgent medical concerns, always recommend calling 911 or visiting the emergency room
- Be patient and understanding with elderly or confused callers
- Confirm all appointment details clearly
- Ask for patient name, date of birth, and contact information for appointments

CLINIC INFORMATION:
- Clinic Name: {self.config.CLINIC_NAME}
- Hours: {self.config.CLINIC_HOURS}
- Available Services: {', '.join(self.services)}

AVAILABLE DOCTORS:
{self._format_doctor_info()}

Remember: You're here to help patients feel comfortable and ensure they get the care they need.
"""

    def _format_doctor_info(self) -> str:
        """Format doctor information for system instructions"""
        doctor_info = []
        for doctor_id, info in self.doctors.items():
            slots = ", ".join(info["available_slots"])
            doctor_info.append(f"- {info['name']} ({info['specialty']}) - Available: {slots}")
        return "\n".join(doctor_info)

    def get_conversation_config(self) -> Dict:
        """Get the conversation configuration for OpenAI Realtime API"""
        return {
            "model": self.config.VOICE_MODEL,
            "voice": self.config.VOICE,
            "instructions": self.get_system_instructions(),
            "input_audio_format": self.config.AUDIO_FORMAT,
            "output_audio_format": self.config.AUDIO_FORMAT,
            "input_audio_transcription": {"model": "whisper-1"},
            "turn_detection": {
                "type": "server_vad",
                "threshold": 0.5,
                "prefix_padding_ms": 300,
                "silence_duration_ms": 500,
            },
            "tools": [
                {
                    "type": "function",
                    "name": "schedule_appointment",
                    "description": "Schedule an appointment for a patient",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "patient_name": {"type": "string", "description": "Patient's full name"},
                            "doctor": {"type": "string", "description": "Requested doctor"},
                            "time_slot": {"type": "string", "description": "Requested time slot"},
                            "reason": {"type": "string", "description": "Reason for visit"},
                            "phone": {"type": "string", "description": "Patient's phone number"},
                        },
                        "required": ["patient_name", "doctor", "time_slot", "reason", "phone"],
                    },
                },
                {
                    "type": "function",
                    "name": "check_doctor_availability",
                    "description": "Check what doctors are available",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "specialty": {"type": "string", "description": "Medical specialty if specified"}
                        },
                    },
                },
            ],
        }

    def handle_function_call(self, function_name: str, arguments: Dict) -> Dict:
        """Handle function calls from the AI"""
        self.logger.info(f"Function called: {function_name} with args: {arguments}")

        if function_name == "schedule_appointment":
            return self._schedule_appointment(**arguments)
        elif function_name == "check_doctor_availability":
            return self._check_doctor_availability(**arguments)
        else:
            return {"error": f"Unknown function: {function_name}"}

    def _schedule_appointment(self, patient_name: str, doctor: str, time_slot: str, reason: str, phone: str) -> Dict:
        """Mock appointment scheduling"""
        # In a real app, this would interact with a scheduling system
        self.logger.info(f"Scheduling appointment for {patient_name} with {doctor} at {time_slot}")

        return {
            "status": "success",
            "message": f"Appointment scheduled for {patient_name} with {doctor} at {time_slot}",
            "confirmation_number": "APT-" + str(hash(f"{patient_name}{time_slot}"))[-6:],
        }

    def _check_doctor_availability(self, specialty: Optional[str] = None) -> Dict:
        """Check doctor availability"""
        available_doctors = []

        for doctor_id, info in self.doctors.items():
            if specialty is None or specialty.lower() in info["specialty"].lower():
                available_doctors.append(
                    {"name": info["name"], "specialty": info["specialty"], "available_slots": info["available_slots"]}
                )

        return {"available_doctors": available_doctors, "total_count": len(available_doctors)}
