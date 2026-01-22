#!/usr/bin/env python3
"""
Enhanced Training Data Generator for DevOps Assistant

This script transforms raw code examples into high-quality, multi-style training data
that teaches the model HOW to communicate, not just WHAT to say.

Key improvements over basic extraction:
1. Multiple response styles (conversational, tutorial, quick-ref, debugging)
2. Explains WHY, not just WHAT
3. Includes best practices and anti-patterns
4. Adds context about when to use each approach
5. Generates negative examples for DPO training
"""

import json
import random
import re
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum


class ResponseStyle(Enum):
    """Different ways the assistant can respond"""
    CONVERSATIONAL = "conversational"  # Like talking to a senior engineer
    TUTORIAL = "tutorial"              # Step-by-step with explanations
    QUICK_REF = "quick_reference"      # Concise, just the essentials
    DEBUGGING = "debugging"            # Troubleshooting focused
    BEST_PRACTICE = "best_practice"    # Focus on production-ready code


class TaskType(Enum):
    """Types of tasks the assistant handles"""
    WRITE_CODE = "write_code"
    EXPLAIN_CODE = "explain_code"
    DEBUG_ERROR = "debug_error"
    REVIEW_CODE = "review_code"
    COMPARE_APPROACHES = "compare_approaches"
    SECURITY_REVIEW = "security_review"
    PERFORMANCE_OPTIMIZE = "performance_optimize"


@dataclass
class TrainingExample:
    """A single training example with metadata"""
    instruction: str
    input: str
    output: str
    style: ResponseStyle = ResponseStyle.CONVERSATIONAL
    task_type: TaskType = TaskType.WRITE_CODE
    domain: str = "python"
    difficulty: str = "intermediate"
    tags: List[str] = field(default_factory=list)

    def to_alpaca(self) -> Dict:
        return {
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output
        }

    def to_sharegpt(self) -> Dict:
        """ShareGPT format for multi-turn conversations"""
        conversations = [
            {"from": "human", "value": self.instruction + ("\n\n" + self.input if self.input else "")},
            {"from": "gpt", "value": self.output}
        ]
        return {"conversations": conversations}


class EnhancedDataGenerator:
    """Generates high-quality training data with multiple styles and approaches"""

    def __init__(self):
        self.instruction_templates = self._load_instruction_templates()
        self.response_templates = self._load_response_templates()
        self.best_practices = self._load_best_practices()
        self.anti_patterns = self._load_anti_patterns()

    def _load_instruction_templates(self) -> Dict[TaskType, List[str]]:
        """Natural ways engineers ask questions"""
        return {
            TaskType.WRITE_CODE: [
                "How do I {task}?",
                "Write a Python function to {task}",
                "I need to {task}, what's the best approach?",
                "Can you show me how to {task} in Python?",
                "What's the cleanest way to {task}?",
                "Help me implement {task}",
                "I'm trying to {task} but not sure where to start",
            ],
            TaskType.EXPLAIN_CODE: [
                "Can you explain what this code does?\n\n```python\n{code}\n```",
                "I found this code but don't understand it:\n\n```python\n{code}\n```",
                "Walk me through this:\n\n```python\n{code}\n```",
                "What's happening in this code?\n\n```python\n{code}\n```",
            ],
            TaskType.DEBUG_ERROR: [
                "I'm getting this error: {error}\n\nMy code:\n```python\n{code}\n```",
                "Why is this failing?\n\n```python\n{code}\n```\n\nError: {error}",
                "This code throws {error}, how do I fix it?",
                "Help me debug this: {error}",
            ],
            TaskType.REVIEW_CODE: [
                "Can you review this code?\n\n```python\n{code}\n```",
                "Is this a good approach?\n\n```python\n{code}\n```",
                "What would you improve here?\n\n```python\n{code}\n```",
                "Any issues with this implementation?\n\n```python\n{code}\n```",
            ],
            TaskType.COMPARE_APPROACHES: [
                "What's the difference between {approach1} and {approach2}?",
                "Should I use {approach1} or {approach2} for {use_case}?",
                "When would I use {approach1} instead of {approach2}?",
            ],
            TaskType.SECURITY_REVIEW: [
                "Is this code secure?\n\n```python\n{code}\n```",
                "Any security issues here?\n\n```python\n{code}\n```",
                "How do I make this more secure?\n\n```python\n{code}\n```",
            ],
            TaskType.PERFORMANCE_OPTIMIZE: [
                "How can I make this faster?\n\n```python\n{code}\n```",
                "This is slow, any optimization ideas?\n\n```python\n{code}\n```",
                "What's the most efficient way to {task}?",
            ],
        }

    def _load_response_templates(self) -> Dict[ResponseStyle, Dict]:
        """Response style templates"""
        return {
            ResponseStyle.CONVERSATIONAL: {
                "opener": [
                    "Good question! ",
                    "Sure thing. ",
                    "Yeah, ",
                    "Ah, ",
                    "",  # Sometimes no opener
                ],
                "explanation_style": "casual",
                "code_intro": [
                    "Here's how I'd do it:",
                    "This should work:",
                    "Try this:",
                    "Here's a clean way to handle that:",
                ],
                "closing": [
                    "",
                    "Let me know if you need anything else.",
                    "Happy to explain any part in more detail.",
                ],
            },
            ResponseStyle.TUTORIAL: {
                "opener": [
                    "Let me walk you through this step by step.\n\n",
                    "I'll break this down into manageable steps.\n\n",
                ],
                "explanation_style": "detailed",
                "use_numbered_steps": True,
                "include_why": True,
            },
            ResponseStyle.QUICK_REF: {
                "opener": [""],
                "explanation_style": "minimal",
                "code_only": True,
            },
            ResponseStyle.DEBUGGING: {
                "opener": [
                    "I see the issue. ",
                    "The problem is ",
                    "This error occurs because ",
                ],
                "include_fix": True,
                "include_prevention": True,
            },
            ResponseStyle.BEST_PRACTICE: {
                "opener": [
                    "For production code, I'd recommend:\n\n",
                    "Here's the production-ready approach:\n\n",
                ],
                "include_error_handling": True,
                "include_logging": True,
                "include_typing": True,
            },
        }

    def _load_best_practices(self) -> Dict[str, List[str]]:
        """Best practices by domain"""
        return {
            "kubernetes": [
                "Always use context managers for API clients",
                "Handle API exceptions gracefully (ApiException)",
                "Use label selectors for targeting resources",
                "Implement proper retry logic for transient failures",
                "Use watch() for real-time updates instead of polling",
            ],
            "docker": [
                "Use multi-stage builds to reduce image size",
                "Don't run containers as root",
                "Pin specific image versions, not 'latest'",
                "Use .dockerignore to exclude unnecessary files",
            ],
            "python": [
                "Use type hints for function signatures",
                "Prefer context managers for resource cleanup",
                "Use pathlib instead of os.path for file operations",
                "Prefer f-strings over .format() or % formatting",
                "Use dataclasses or named tuples for structured data",
            ],
            "terraform": [
                "Use modules for reusable infrastructure",
                "Store state remotely with locking",
                "Use variables for environment-specific values",
                "Implement proper tagging strategy",
            ],
            "security": [
                "Never hardcode credentials",
                "Use environment variables or secret managers",
                "Validate and sanitize all user input",
                "Use parameterized queries to prevent injection",
            ],
        }

    def _load_anti_patterns(self) -> Dict[str, List[Dict]]:
        """Anti-patterns with explanations - for negative examples"""
        return {
            "kubernetes": [
                {
                    "bad": "client.CoreV1Api().list_pod_for_all_namespaces()",
                    "why_bad": "Creates new client on every call, wastes resources",
                    "good": "v1 = client.CoreV1Api()\nv1.list_pod_for_all_namespaces()",
                    "why_good": "Reuses client connection",
                },
                {
                    "bad": "while True:\n    pods = v1.list_pod_for_all_namespaces()\n    time.sleep(1)",
                    "why_bad": "Polling is inefficient and can overwhelm the API",
                    "good": "w = watch.Watch()\nfor event in w.stream(v1.list_pod_for_all_namespaces):",
                    "why_good": "Uses watch for efficient real-time updates",
                },
            ],
            "python": [
                {
                    "bad": "except Exception:\n    pass",
                    "why_bad": "Silently swallows all errors, makes debugging impossible",
                    "good": "except SpecificException as e:\n    logger.error(f'Failed: {e}')\n    raise",
                    "why_good": "Catches specific exceptions, logs them, re-raises",
                },
                {
                    "bad": "f = open('file.txt')\ndata = f.read()\nf.close()",
                    "why_bad": "If read() fails, file is never closed",
                    "good": "with open('file.txt') as f:\n    data = f.read()",
                    "why_good": "Context manager ensures cleanup even on exceptions",
                },
            ],
            "security": [
                {
                    "bad": "password = 'hardcoded_secret_123'",
                    "why_bad": "Secrets in code get committed to version control",
                    "good": "password = os.environ.get('DB_PASSWORD')",
                    "why_good": "Secrets come from environment, not code",
                },
                {
                    "bad": 'query = f"SELECT * FROM users WHERE id = {user_id}"',
                    "why_bad": "SQL injection vulnerability",
                    "good": 'cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))',
                    "why_good": "Parameterized query prevents injection",
                },
            ],
        }

    def generate_conversational_response(
        self,
        code: str,
        task_description: str,
        domain: str = "python"
    ) -> str:
        """Generate a response as if talking to a senior engineer"""

        template = self.response_templates[ResponseStyle.CONVERSATIONAL]
        opener = random.choice(template["opener"])
        code_intro = random.choice(template["code_intro"])
        closing = random.choice(template["closing"])

        # Build response
        response_parts = [opener]

        # Add context about the approach
        response_parts.append(f"{code_intro}\n\n```python\n{code}\n```")

        # Add explanation of key parts
        if random.random() > 0.3:  # 70% of the time add explanation
            response_parts.append(self._generate_code_explanation(code, domain))

        # Add relevant best practice
        if domain in self.best_practices and random.random() > 0.5:
            practice = random.choice(self.best_practices[domain])
            response_parts.append(f"\n\n**Tip:** {practice}")

        if closing:
            response_parts.append(f"\n\n{closing}")

        return "".join(response_parts)

    def generate_tutorial_response(
        self,
        code: str,
        task_description: str,
        domain: str = "python"
    ) -> str:
        """Generate step-by-step tutorial style response"""

        response_parts = ["Let me walk you through this.\n\n"]

        # Parse code into logical sections
        sections = self._parse_code_sections(code)

        for i, (section_name, section_code) in enumerate(sections, 1):
            response_parts.append(f"**Step {i}: {section_name}**\n\n")
            response_parts.append(f"```python\n{section_code}\n```\n\n")
            response_parts.append(self._explain_section(section_code, domain))
            response_parts.append("\n\n")

        # Add complete example
        response_parts.append("**Complete code:**\n\n")
        response_parts.append(f"```python\n{code}\n```")

        return "".join(response_parts)

    def generate_debugging_response(
        self,
        error: str,
        code: str,
        fix: str,
        domain: str = "python"
    ) -> str:
        """Generate a debugging/troubleshooting response"""

        response_parts = []

        # Identify the problem
        response_parts.append(f"The error `{error}` occurs because ")
        response_parts.append(self._explain_error(error, domain))
        response_parts.append("\n\n")

        # Show the problematic code
        response_parts.append("**The issue:**\n\n")
        response_parts.append(f"```python\n{code}\n```\n\n")

        # Provide the fix
        response_parts.append("**The fix:**\n\n")
        response_parts.append(f"```python\n{fix}\n```\n\n")

        # Explain why
        response_parts.append("**Why this works:** ")
        response_parts.append(self._explain_fix(error, fix, domain))

        # Prevention tips
        response_parts.append("\n\n**To prevent this in the future:**\n")
        response_parts.append(self._prevention_tips(error, domain))

        return "".join(response_parts)

    def generate_comparison_response(
        self,
        approach1: str,
        approach2: str,
        code1: str,
        code2: str,
        domain: str = "python"
    ) -> str:
        """Generate a comparison between two approaches"""

        response_parts = [f"Both `{approach1}` and `{approach2}` have their place. Here's when to use each:\n\n"]

        # Approach 1
        response_parts.append(f"## {approach1}\n\n")
        response_parts.append(f"```python\n{code1}\n```\n\n")
        response_parts.append("**Use when:**\n")
        response_parts.append(self._when_to_use(approach1, domain))
        response_parts.append("\n\n")

        # Approach 2
        response_parts.append(f"## {approach2}\n\n")
        response_parts.append(f"```python\n{code2}\n```\n\n")
        response_parts.append("**Use when:**\n")
        response_parts.append(self._when_to_use(approach2, domain))
        response_parts.append("\n\n")

        # Summary
        response_parts.append("## Quick decision guide\n\n")
        response_parts.append(self._comparison_summary(approach1, approach2, domain))

        return "".join(response_parts)

    def generate_dpo_pair(
        self,
        instruction: str,
        domain: str = "python"
    ) -> Tuple[Dict, Dict]:
        """Generate a chosen/rejected pair for DPO training"""

        if domain not in self.anti_patterns:
            return None

        anti_pattern = random.choice(self.anti_patterns[domain])

        # Rejected response (the bad way)
        rejected = {
            "instruction": instruction,
            "input": "",
            "output": f"Here you go:\n\n```python\n{anti_pattern['bad']}\n```"
        }

        # Chosen response (the good way with explanation)
        chosen = {
            "instruction": instruction,
            "input": "",
            "output": (
                f"Here's the recommended approach:\n\n"
                f"```python\n{anti_pattern['good']}\n```\n\n"
                f"**Why this approach:** {anti_pattern['why_good']}\n\n"
                f"**Avoid this pattern:**\n```python\n{anti_pattern['bad']}\n```\n"
                f"This is problematic because: {anti_pattern['why_bad']}"
            )
        }

        return chosen, rejected

    def _parse_code_sections(self, code: str) -> List[Tuple[str, str]]:
        """Parse code into logical sections for tutorial"""
        sections = []

        # Simple heuristic: split by function definitions and imports
        lines = code.split('\n')
        current_section = []
        current_name = "Setup"

        for line in lines:
            if line.startswith('import ') or line.startswith('from '):
                if current_section and current_name != "Imports":
                    sections.append((current_name, '\n'.join(current_section)))
                    current_section = []
                current_name = "Imports"
            elif line.startswith('def '):
                if current_section:
                    sections.append((current_name, '\n'.join(current_section)))
                    current_section = []
                # Extract function name
                match = re.match(r'def (\w+)', line)
                current_name = f"Define {match.group(1)}()" if match else "Define function"
            elif line.startswith('class '):
                if current_section:
                    sections.append((current_name, '\n'.join(current_section)))
                    current_section = []
                match = re.match(r'class (\w+)', line)
                current_name = f"Define {match.group(1)} class" if match else "Define class"

            current_section.append(line)

        if current_section:
            sections.append((current_name, '\n'.join(current_section)))

        return sections if sections else [("Implementation", code)]

    def _generate_code_explanation(self, code: str, domain: str) -> str:
        """Generate explanation for code"""
        explanations = []

        # Check for common patterns
        if 'with ' in code:
            explanations.append("Using a context manager ensures proper cleanup even if an error occurs.")
        if 'try:' in code:
            explanations.append("The try/except block handles potential errors gracefully.")
        if '@' in code:
            explanations.append("The decorator modifies the function's behavior without changing its code.")
        if 'async ' in code or 'await ' in code:
            explanations.append("This uses async/await for non-blocking I/O operations.")
        if 'yield' in code:
            explanations.append("Using a generator here is memory-efficient for large datasets.")

        if explanations:
            return "\n\n**What's happening here:** " + " ".join(explanations)
        return ""

    def _explain_section(self, code: str, domain: str) -> str:
        """Explain a section of code"""
        # This would be enhanced with actual code analysis
        return "This sets up the necessary components for our task."

    def _explain_error(self, error: str, domain: str) -> str:
        """Explain why an error occurs"""
        error_explanations = {
            "ApiException": "the Kubernetes API rejected the request",
            "ConnectionError": "the service couldn't be reached",
            "TimeoutError": "the operation took too long",
            "KeyError": "the expected key doesn't exist in the dictionary",
            "AttributeError": "you're trying to access an attribute that doesn't exist",
            "TypeError": "the types don't match what the function expects",
            "ImportError": "the module isn't installed or can't be found",
        }

        for error_type, explanation in error_explanations.items():
            if error_type in error:
                return explanation

        return "there's a mismatch between what the code expects and what it received"

    def _explain_fix(self, error: str, fix: str, domain: str) -> str:
        """Explain why a fix works"""
        return "This addresses the root cause by ensuring proper handling of the edge case."

    def _prevention_tips(self, error: str, domain: str) -> str:
        """Generate tips to prevent similar errors"""
        tips = [
            "- Add type hints to catch type mismatches early",
            "- Use linting tools like pylint or ruff",
            "- Write unit tests for edge cases",
            "- Use proper error handling from the start",
        ]
        return '\n'.join(tips[:3])

    def _when_to_use(self, approach: str, domain: str) -> str:
        """Generate use cases for an approach"""
        return "- When you need flexibility\n- In production environments\n- When working with large datasets"

    def _comparison_summary(self, approach1: str, approach2: str, domain: str) -> str:
        """Generate comparison summary"""
        return f"| Criteria | {approach1} | {approach2} |\n|----------|-------------|-------------|\n| Speed | Fast | Moderate |\n| Memory | Low | Higher |\n| Readability | Good | Better |"

    def transform_basic_example(
        self,
        basic_example: Dict,
        style: ResponseStyle = None,
        domain: str = "python"
    ) -> List[TrainingExample]:
        """Transform a basic code example into multiple high-quality examples"""

        examples = []
        instruction = basic_example.get("instruction", "")
        code = basic_example.get("output", "")

        if not style:
            # Generate multiple styles
            styles = [ResponseStyle.CONVERSATIONAL, ResponseStyle.TUTORIAL, ResponseStyle.QUICK_REF]
        else:
            styles = [style]

        for s in styles:
            if s == ResponseStyle.CONVERSATIONAL:
                output = self.generate_conversational_response(code, instruction, domain)
            elif s == ResponseStyle.TUTORIAL:
                output = self.generate_tutorial_response(code, instruction, domain)
            elif s == ResponseStyle.QUICK_REF:
                output = f"```python\n{code}\n```"
            else:
                output = self.generate_conversational_response(code, instruction, domain)

            # Vary the instruction phrasing
            varied_instruction = self._vary_instruction(instruction)

            examples.append(TrainingExample(
                instruction=varied_instruction,
                input="",
                output=output,
                style=s,
                domain=domain,
            ))

        return examples

    def _vary_instruction(self, instruction: str) -> str:
        """Create natural variations of instructions"""
        # Extract the core task
        task = instruction.lower()

        # Remove common prefixes
        prefixes_to_remove = [
            "write python code that ",
            "create a python function called ",
            "show me how to ",
            "how do i ",
            "write a python example for ",
        ]

        for prefix in prefixes_to_remove:
            if task.startswith(prefix):
                task = task[len(prefix):]
                break

        # Pick a natural phrasing
        templates = [
            f"How do I {task}?",
            f"What's the best way to {task}?",
            f"Can you show me how to {task}?",
            f"I need to {task}",
            f"Help me {task}",
        ]

        return random.choice(templates)


def process_existing_dataset(
    input_path: str,
    output_path: str,
    domain: str = "python",
    include_dpo: bool = True
) -> None:
    """Process existing dataset to enhance quality"""

    generator = EnhancedDataGenerator()

    with open(input_path, 'r') as f:
        raw_data = json.load(f)

    enhanced_data = []
    dpo_pairs = []

    for example in raw_data:
        # Generate multiple styles for each example
        transformed = generator.transform_basic_example(example, domain=domain)
        enhanced_data.extend([t.to_alpaca() for t in transformed])

        # Generate DPO pairs
        if include_dpo:
            dpo_pair = generator.generate_dpo_pair(
                example.get("instruction", ""),
                domain=domain
            )
            if dpo_pair:
                dpo_pairs.append(dpo_pair)

    # Deduplicate
    seen = set()
    deduped = []
    for item in enhanced_data:
        key = hashlib.md5(json.dumps(item, sort_keys=True).encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            deduped.append(item)

    # Save enhanced data
    with open(output_path, 'w') as f:
        json.dump(deduped, f, indent=2)

    print(f"Generated {len(deduped)} enhanced examples from {len(raw_data)} originals")

    # Save DPO pairs if generated
    if dpo_pairs:
        dpo_path = output_path.replace('.json', '_dpo.json')
        with open(dpo_path, 'w') as f:
            json.dump(dpo_pairs, f, indent=2)
        print(f"Generated {len(dpo_pairs)} DPO training pairs")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Enhance training data quality")
    parser.add_argument("--input", required=True, help="Input dataset path")
    parser.add_argument("--output", required=True, help="Output dataset path")
    parser.add_argument("--domain", default="python", help="Domain (python, kubernetes, etc)")
    parser.add_argument("--no-dpo", action="store_true", help="Skip DPO pair generation")

    args = parser.parse_args()

    process_existing_dataset(
        args.input,
        args.output,
        domain=args.domain,
        include_dpo=not args.no_dpo
    )
