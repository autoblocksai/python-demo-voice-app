import json
import os

from autoblocks.testing.models import BaseTestEvaluator
from autoblocks.testing.models import Evaluation
from autoblocks.testing.models import Threshold
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Types will be resolved at runtime

load_dotenv()

# Initialize async OpenAI client
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# Base evaluator with shared logic for voice conversations
class BaseVoiceConversationEvaluator(BaseTestEvaluator):
    def __init__(self, criterion_key: str, criterion_description: str):
        self.criterion_key = criterion_key
        self.criterion_description = criterion_description

    async def evaluate_test_case(self, test_case, output) -> Evaluation:
        # Format the conversation for evaluation
        conversation_text = ""
        for msg in output.messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if role == "user":
                conversation_text += f"PATIENT: {content}\n"
            elif role == "assistant":
                conversation_text += f"RECEPTIONIST: {content}\n"

        # Add voice conversation context
        context_info = f"""
Voice Conversation Context:
- Patient Voice: {test_case.patient_voice}
- Total Turns: {output.total_turns}
- Conversation Duration: {output.conversation_duration_seconds:.1f} seconds
- Receptionist Responses: {output.receptionist_responses_count}
- Scenario: {test_case.scenario_id}
"""

        # Define the evaluation function schema for this specific criterion
        evaluation_function = {
            "name": f"evaluate_{self.criterion_key}",
            "description": f"Evaluate {self.criterion_key} in a voice-based medical receptionist conversation",
            "parameters": {
                "type": "object",
                "properties": {
                    self.criterion_key: {
                        "type": "string",
                        "enum": ["poor", "fair", "good", "excellent"],
                        "description": self.criterion_description,
                    },
                    "reason": {
                        "type": "string",
                        "description": f"Provide a brief explanation for why you rated {self.criterion_key} as you did, including specific examples from the conversation",  # noqa: E501
                    },
                },
                "required": [self.criterion_key, "reason"],
            },
        }

        # Create evaluation prompt
        evaluation_prompt = f"""
You are an expert evaluator of voice-based medical receptionist conversations.
Please evaluate the following conversation focusing specifically on {self.criterion_key}.

{context_info}

Criterion: {self.criterion_description}

Conversation:
{conversation_text}

Please evaluate this specific criterion and call the evaluation function with your assessment.
Consider that this is a voice conversation.
Evaluate based on how natural and appropriate the interaction would be over the phone.
"""

        try:
            # Call OpenAI API with function calling (async)
            response = await openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": evaluation_prompt}],
                tools=[{"type": "function", "function": evaluation_function}],
                tool_choice={"type": "function", "function": {"name": f"evaluate_{self.criterion_key}"}},
                temperature=0.1,
            )

            # Parse the function call response
            function_call = response.choices[0].message.tool_calls[0].function
            evaluation_data = json.loads(function_call.arguments)

            # Map evaluation categories to scores
            score_mapping = {"poor": 0, "fair": 0.25, "good": 0.75, "excellent": 1.0}

            # Get the score for this specific criterion
            criterion_value = evaluation_data.get(self.criterion_key, "fair")
            score = score_mapping.get(criterion_value, 0)
            reason = evaluation_data.get("reason", "No reason provided")

        except Exception as e:
            print(f"Error in LLM evaluation for {self.criterion_key}: {e}")
            score = 0
            reason = f"Evaluation failed: {str(e)}"

        return Evaluation(
            score=score,
            threshold=Threshold(
                gte=0.75,
            ),
            metadata={"reason": reason},
        )


class VoiceNaturalness(BaseVoiceConversationEvaluator):
    id = "voice_naturalness"

    def __init__(self):
        super().__init__(
            "voice_naturalness", "How natural and conversational did the receptionist sound in a voice interaction?"
        )


class MedicalProfessionalism(BaseVoiceConversationEvaluator):
    id = "medical_professionalism"

    def __init__(self):
        super().__init__(
            "medical_professionalism",
            "Was the receptionist appropriately professional for a medical setting while maintaining warmth?",
        )


class AppointmentHandling(BaseVoiceConversationEvaluator):
    id = "appointment_handling"

    def __init__(self):
        super().__init__(
            "appointment_handling",
            "How effectively did the receptionist handle appointment scheduling, rescheduling, or related requests?",
        )


class PatientEmpathy(BaseVoiceConversationEvaluator):
    id = "patient_empathy"

    def __init__(self):
        super().__init__(
            "patient_empathy",
            "Did the receptionist show appropriate empathy and understanding for the patient's concerns?",
        )


class InformationGathering(BaseVoiceConversationEvaluator):
    id = "information_gathering"

    def __init__(self):
        super().__init__(
            "information_gathering", "How well did the receptionist gather necessary information from the patient?"
        )


class ConversationFlow(BaseVoiceConversationEvaluator):
    id = "conversation_flow"

    def __init__(self):
        super().__init__(
            "conversation_flow", "Did the conversation flow naturally with appropriate transitions and responses?"
        )


class UrgencyAssessment(BaseVoiceConversationEvaluator):
    id = "urgency_assessment"

    def __init__(self):
        super().__init__(
            "urgency_assessment",
            "Did the receptionist appropriately assess and respond to the urgency of the patient's needs?",
        )


class CallResolution(BaseVoiceConversationEvaluator):
    id = "call_resolution"

    def __init__(self):
        super().__init__("call_resolution", "Was the patient's call resolved satisfactorily with clear next steps?")
