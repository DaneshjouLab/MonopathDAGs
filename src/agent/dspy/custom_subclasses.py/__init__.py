from dspy.adapters.json_adapter import JSONAdapter
from dspy.signatures.signature import Signature
from dspy.adapters.chat_adapter import FieldInfoWithName
from typing import Type, Any, Dict
import json
from dspy.adapters.utils import format_field_value, serialize_for_json

class CustomJSONAdapter(JSONAdapter):
    def format_field_structure(self, signature: Type[Signature]) -> str:
        # Completely removes the "Inputs will have..." boilerplate
        return ""

    def format_task_description(self, signature: Type[Signature]) -> str:
        # Just returns your Signature.__doc__
        return signature.instructions

    def user_message_output_requirements(self, signature: Type[Signature]) -> str:
        # Optional: remove DSPy’s "respond with..." reminders
        return ""

    def format_field_with_value(self, fields_with_values: Dict[FieldInfoWithName, Any], role: str = "user") -> str:
        if role == "user":
            output = []
            for field, value in fields_with_values.items():
                # Pull schema_desc if available, otherwise fall back to desc
                schema_desc = field.info.json_schema_extra.get("schema_desc")
                if schema_desc:
                    output.append(f"// {schema_desc}")  # Comment-style hint
                formatted = format_field_value(field_info=field.info, value=value)
                output.append(f"[[ ## {field.name} ## ]]\n{formatted}")
            return "\n\n".join(output).strip()
        else:
            # Assistant JSON-formatted output
            return json.dumps(
                serialize_for_json({f.name: v for f, v in fields_with_values.items()}),
                indent=2
            )
