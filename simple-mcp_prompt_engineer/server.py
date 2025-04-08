from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
import json
import re
from enum import Enum
from datetime import datetime
from mcp.server.fastmcp import FastMCP
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.console import Group
from rich.style import Style

console = Console(stderr=True)

class OptimizationStage(Enum):
    INITIAL_ANALYSIS = "Initial Analysis"
    RULES_APPLICATION = "Rules Application" 
    STRUCTURING = "Structuring"
    VERIFICATION = "Verification"
    REFINEMENT = "Refinement"
    FINAL = "Final"

@dataclass
class PromptRule:
    """Rule for prompt optimization with implementation."""
    name: str
    description: str
    template_instruction: str  # The exact text to show in the template's <rules> section
    apply: callable  # Function that applies the rule to a prompt

@dataclass
class PromptData:
    original_prompt: str
    optimized_prompt: str
    version: int = 1
    stage: OptimizationStage = OptimizationStage.INITIAL_ANALYSIS
    rules_applied: List[str] = field(default_factory=list)
    user_feedback: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class OptimizationHistory:
    history: List[PromptData] = field(default_factory=list)
    
    def add_version(self, prompt_data: PromptData) -> None:
        """Add a new version to the history."""
        self.history.append(prompt_data)
    
    def get_latest(self) -> Optional[PromptData]:
        """Get the latest prompt version."""
        if not self.history:
            return None
        return self.history[-1]
    
    def get_version(self, version: int) -> Optional[PromptData]:
        """Get a specific prompt version."""
        for prompt in self.history:
            if prompt.version == version:
                return prompt
        return None
    
    def clear(self) -> None:
        """Clear all history."""
        self.history.clear()

class PromptType(Enum):
    RESEARCH = "Research"
    CREATIVE = "Creative"
    TECHNICAL = "Technical"
    ANALYTICAL = "Analytical"
    GENERAL = "General"

class PromptOptimizationEngine:
    """Engine for automatically optimizing prompts based on rules."""
    
    def __init__(self):
        # Define optimization rules - easily editable
        self.rules = [
            PromptRule(
                name="add_steps_section",
                description="Add steps section",
                template_instruction="The prompt must include a 'steps' section that guides through the optimal completion process",
                apply=self._add_steps_section
            ),
            PromptRule(
                name="improve_structure",
                description="Improve structure",
                template_instruction="The prompt should be structured with clear sections for better organization",
                apply=self._improve_structure
            ),
            PromptRule(
                name="add_context_preservation",
                description="Add context preservation",
                template_instruction="Ensure all important context from the original is preserved and highlighted",
                apply=self._add_context_preservation
            ),
            PromptRule(
                name="add_clarity_guidelines",
                description="Add clarity guidelines",
                template_instruction="Include guidelines for clear, specific output formatting",
                apply=self._add_clarity_guidelines
            ),
            PromptRule(
                name="remove_redundancy",
                description="Remove redundancy",
                template_instruction="Eliminate redundant or repetitive elements while preserving core meaning",
                apply=self._remove_redundancy
            )
        ]
        
        # Define notes for different prompt types - easily editable in plain language
        self.prompt_type_notes = {
            PromptType.RESEARCH: [
                "ALWAYS RETURN AN ARTIFACT",
                "The prompt is for research purposes and should emphasize depth and credibility",
                "Use Brave Search to verify factual accuracy of provided information",
                "Use Brave Search to get an idea of the research topic",
                "Consider comparative analysis between different perspectives",
                "Prioritize scholarly sources and evidence-based information"
            ],
            PromptType.CREATIVE: [
                "ALWAYS RETURN AN ARTIFACT",
                "The prompt is for creative content generation",
                "Focus on originality and engaging narrative elements",
                "Consider emotional impact and reader engagement",
                "Provide clear stylistic direction"
            ],
            PromptType.TECHNICAL: [
                "ALWAYS RETURN AN ARTIFACT",
                "The prompt is for technical content that requires precision",
                "Ensure technical terminology is correctly specified",
                "Focus on step-by-step clarity and logical progression",
                "Consider including validation criteria for the output"
            ],
            PromptType.ANALYTICAL: [
                "ALWAYS RETURN AN ARTIFACT",
                "The prompt is for analytical investigation",
                "Emphasize logical frameworks and evaluation criteria",
                "Consider multiple analytical perspectives",
                "Focus on evidence-based conclusions"
            ],
            PromptType.GENERAL: [
                "ALWAYS RETURN AN ARTIFACT",
                "The prompt is for general purpose content",
                "Focus on clarity and specificity in the request",
                "Consider adding examples of desired output format",
                "Ensure all requirements are explicitly stated"
            ]
        }
        
        # Hardcoded tools section
        self.tools_section = [
            "Use Brave Search to verify factual assumptions and current information",
            "Use Sequential Thinking to ensure logical flow and complete coverage of the topic"
        ]
    
    def _add_steps_section(self, prompt: str) -> str:
        """Add a steps section to the prompt if it doesn't have one."""
        if "<steps>" in prompt.lower() or "## steps" in prompt.lower():
            return prompt
        
        # Analyze prompt to determine appropriate steps
        task_type = self._determine_task_type(prompt)
        steps = self._generate_steps_for_task(task_type, prompt)
        
        # Add steps section
        return f"{prompt}\n\n## Steps to Complete the Task\n{steps}"
    
    def _improve_structure(self, prompt: str) -> str:
        """Improve the structure with clear sections."""
        if "##" in prompt or "<section>" in prompt:
            return prompt  # Already has structure
        
        sections = {
            "Context": "Understand the background and requirements",
            "Requirements": "Key elements that must be addressed",
            "Constraints": "Limitations to work within",
            "Output Format": "How the final output should be structured"
        }
        
        # Extract or infer content for each section from the prompt
        structured_prompt = "# Optimized Task\n\n"
        for section, description in sections.items():
            content = self._extract_or_infer_section(prompt, section)
            structured_prompt += f"## {section}\n{content}\n\n"
        
        # Add original prompt at the end for reference
        structured_prompt += f"## Original Request\n{prompt}"
        
        return structured_prompt
    
    def _add_context_preservation(self, prompt: str) -> str:
        """Add guidelines to preserve important context."""
        if "preserve context" in prompt.lower() or "maintain context" in prompt.lower():
            return prompt
        
        preservation_note = "\n\n## Important Context\nEnsure all key information from the original request is preserved in your response."
        
        # Extract key terms to emphasize
        key_terms = self._extract_key_terms(prompt)
        if key_terms:
            preservation_note += f"\nPay particular attention to: {', '.join(key_terms)}."
        
        return prompt + preservation_note
    
    def _add_clarity_guidelines(self, prompt: str) -> str:
        """Add guidelines for clarity in outputs."""
        if "be clear" in prompt.lower() or "clarity" in prompt.lower():
            return prompt
        
        clarity_guidelines = "\n\n## Clarity Guidelines\n"
        clarity_guidelines += "- Use precise, specific language\n"
        clarity_guidelines += "- Define technical terms when first introduced\n"
        clarity_guidelines += "- Use examples to illustrate complex concepts\n"
        clarity_guidelines += "- Break down complex ideas into smaller components\n"
        
        return prompt + clarity_guidelines
    
    def _remove_redundancy(self, prompt: str) -> str:
        """Remove redundant or repetitive elements."""
        # Split into lines and remove duplicates while preserving order
        lines = prompt.split("\n")
        seen = set()
        unique_lines = []
        
        for line in lines:
            # Normalize for comparison (lowercase, strip punctuation)
            norm_line = re.sub(r'[^\w\s]', '', line.lower().strip())
            if norm_line and norm_line not in seen and not self._is_similar_to_any(norm_line, seen):
                seen.add(norm_line)
                unique_lines.append(line)
            elif not norm_line:  # Keep empty lines for structure
                unique_lines.append(line)
        
        return "\n".join(unique_lines)
    
    # Helper methods
    def _determine_task_type(self, prompt: str) -> str:
        """Determine the type of task in the prompt."""
        prompt_lower = prompt.lower()
        if "analyze" in prompt_lower or "analysis" in prompt_lower:
            return "analysis"
        elif "generate" in prompt_lower or "create" in prompt_lower:
            return "generation"
        elif "summarize" in prompt_lower or "summary" in prompt_lower:
            return "summarization"
        elif "compare" in prompt_lower or "contrast" in prompt_lower:
            return "comparison"
        elif "explain" in prompt_lower or "description" in prompt_lower:
            return "explanation"
        else:
            return "general"
    
    def _generate_steps_for_task(self, task_type: str, prompt: str) -> str:
        """Generate appropriate steps based on the task type."""
        steps = ""
        
        if task_type == "analysis":
            steps = "1. Identify key elements for analysis\n"
            steps += "2. Gather relevant information\n"
            steps += "3. Apply analytical framework\n"
            steps += "4. Draw insights from patterns\n"
            steps += "5. Formulate conclusions based on evidence"
        elif task_type == "generation":
            steps = "1. Understand requirements and constraints\n"
            steps += "2. Brainstorm potential approaches\n"
            steps += "3. Outline the structure\n"
            steps += "4. Develop detailed content\n"
            steps += "5. Review and refine for quality"
        elif task_type == "summarization":
            steps = "1. Identify key information and main points\n"
            steps += "2. Organize information by importance\n"
            steps += "3. Condense while maintaining meaning\n"
            steps += "4. Ensure accuracy and completeness\n"
            steps += "5. Present in clear, concise language"
        elif task_type == "comparison":
            steps = "1. Identify elements to be compared\n"
            steps += "2. Determine relevant criteria for comparison\n"
            steps += "3. Analyze similarities systematically\n"
            steps += "4. Analyze differences systematically\n"
            steps += "5. Draw meaningful conclusions from the comparison"
        elif task_type == "explanation":
            steps = "1. Identify core concepts to explain\n"
            steps += "2. Consider the appropriate level of detail\n"
            steps += "3. Use clear definitions and examples\n"
            steps += "4. Connect concepts logically\n"
            steps += "5. Check for clarity and completeness"
        else:  # General default steps
            steps = "1. Understand the full context of the request\n"
            steps += "2. Identify key requirements and constraints\n"
            steps += "3. Plan the approach to address all elements\n"
            steps += "4. Execute the plan thoroughly\n"
            steps += "5. Review for completeness and quality"
        
        return steps
    
    def _extract_or_infer_section(self, prompt: str, section_name: str) -> str:
        """Extract content for a section or infer it if not explicitly present."""
        # Simple inference rules for demonstration
        if section_name == "Context":
            # Look for background information, often at the beginning
            first_paragraph = prompt.split("\n\n")[0] if "\n\n" in prompt else prompt
            return first_paragraph
        elif section_name == "Requirements":
            # Look for action words and requests
            requirements = []
            for line in prompt.split("\n"):
                if any(word in line.lower() for word in ["need", "must", "should", "require"]):
                    requirements.append(line)
            return "\n".join(requirements) if requirements else "Address all elements specified in the request."
        elif section_name == "Constraints":
            # Look for limitations or restrictions
            constraints = []
            for line in prompt.split("\n"):
                if any(word in line.lower() for word in ["limit", "constraint", "restriction", "cannot", "don't"]):
                    constraints.append(line)
            return "\n".join(constraints) if constraints else "No explicit constraints specified."
        elif section_name == "Output Format":
            # Look for format specifications
            if "format" in prompt.lower():
                for line in prompt.split("\n"):
                    if "format" in line.lower():
                        return line
            return "Present the response in a clear, well-structured format appropriate to the task."
        return f"[Content for {section_name} to be determined]"
    
    def _extract_key_terms(self, prompt: str) -> List[str]:
        """Extract likely key terms from the prompt."""
        # Very basic extraction for demonstration
        words = re.findall(r'\b[A-Z][a-z]*\b', prompt)  # Capitalized words
        return list(set(words))[:5]  # Return up to 5 unique terms
    
    def _is_similar_to_any(self, line: str, seen_lines: set) -> bool:
        """Check if a line is very similar to any in the already seen lines."""
        for seen in seen_lines:
            if self._similarity(line, seen) > 0.8:  # Arbitrary threshold
                return True
        return False
    
    def _similarity(self, a: str, b: str) -> float:
        """Calculate a simple similarity score between two strings."""
        if not a or not b:
            return 0.0
        
        # Very basic similarity - shared words divided by total unique words
        words_a = set(a.split())
        words_b = set(b.split())
        
        if not words_a or not words_b:
            return 0.0
        
        intersection = len(words_a.intersection(words_b))
        union = len(words_a.union(words_b))
        
        return intersection / union if union > 0 else 0.0
    
    def detect_prompt_type(self, prompt: str) -> PromptType:
        """Detect the type of prompt for appropriate notes selection."""
        prompt_lower = prompt.lower()
        
        # Look for research indicators
        if any(word in prompt_lower for word in ["research", "study", "investigate", "analyze", "evidence"]):
            return PromptType.RESEARCH
            
        # Look for creative indicators
        elif any(word in prompt_lower for word in ["create", "story", "write", "imagine", "creative"]):
            return PromptType.CREATIVE
            
        # Look for technical indicators
        elif any(word in prompt_lower for word in ["code", "program", "technical", "algorithm", "system"]):
            return PromptType.TECHNICAL
            
        # Look for analytical indicators
        elif any(word in prompt_lower for word in ["analyze", "compare", "evaluate", "assess"]):
            return PromptType.ANALYTICAL
            
        # Default to general
        else:
            return PromptType.GENERAL
    
    def optimize_prompt(self, prompt: str) -> Dict[str, Any]:
        """Apply optimization rules to improve a prompt."""
        optimized = prompt
        applied_rules = []
        
        # Apply all rules and track which ones actually changed the prompt
        for rule in self.rules:
            before = optimized
            optimized = rule.apply(optimized)
            if optimized != before:
                applied_rules.append(rule.name)
        
        return {
            "optimized_prompt": optimized,
            "rules_applied": applied_rules,
            "prompt_type": self.detect_prompt_type(prompt)
        }

class PromptEngineeringServer:
    def __init__(self):
        self.optimization_history = OptimizationHistory()
        self.optimization_engine = PromptOptimizationEngine()
        
        # The base template - notes and rules will be inserted dynamically
        self.base_template = """Your task for today is to perform prompt engineering on a prompt. The next message will be divided into XML tags.
* <rules>: Describes the rules you must follow to improve the prompt
* <prompt>: The actual prompt
* <notes>: Any comments that come to mind
* <tools>: The tools you will use for redefining the prompt.

<rules>
{rules}
</rules>

<notes>
{notes}
</notes>

<tools>
{tools}
</tools>

<prompt>
{prompt}
</prompt>
"""
    
    def optimize_prompt(self, prompt: str) -> str:
        """Optimize a prompt and return only the optimized version."""
        try:
            # Apply optimization rules
            optimization_result = self.optimization_engine.optimize_prompt(prompt)
            optimized_prompt = optimization_result["optimized_prompt"]
            applied_rule_names = optimization_result["rules_applied"]
            prompt_type = optimization_result["prompt_type"]
            
            # Get the complete rule instructions for the rules that were applied
            rules_instructions = []
            for rule_name in applied_rule_names:
                for rule in self.optimization_engine.rules:
                    if rule.name == rule_name:
                        rules_instructions.append(f"* {rule.template_instruction}")
            
            # If no rules were applied, use a default instruction
            if not rules_instructions:
                rules_instructions = ["* Ensure the prompt is clear, specific, and well-structured"]
            
            # Get notes for the detected prompt type
            type_notes = self.optimization_engine.prompt_type_notes.get(prompt_type, 
                                                                        self.optimization_engine.prompt_type_notes[PromptType.GENERAL])
            notes_formatted = "\n".join(f"* {note}" for note in type_notes)
            
            # Format tools section
            tools_formatted = "\n".join(f"* {tool}" for tool in self.optimization_engine.tools_section)
            
            # Create the final template
            final_prompt = self.base_template.format(
                rules="\n".join(rules_instructions),
                notes=notes_formatted,
                tools=tools_formatted,
                prompt=optimized_prompt
            )
            
            # Create prompt data for history
            latest = self.optimization_history.get_latest()
            version = (latest.version + 1) if latest else 1
            
            prompt_data = PromptData(
                original_prompt=prompt,
                optimized_prompt=final_prompt,
                version=version,
                stage=OptimizationStage.RULES_APPLICATION,
                rules_applied=applied_rule_names
            )
            
            # Add to history
            self.optimization_history.add_version(prompt_data)
            
            # Log output for server monitoring
            console.print(f"[bold green]Optimized prompt v{version}[/bold green]")
            console.print(f"Applied rules: {', '.join(applied_rule_names)}")
            console.print(f"Prompt type: {prompt_type.value}")
            
            # Return only the optimized prompt
            return final_prompt
            
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")
            return f"Error optimizing prompt: {str(e)}"
    
    def refine_prompt(self, feedback: str) -> str:
        """Refine a prompt based on user feedback."""
        try:
            latest = self.optimization_history.get_latest()
            if not latest:
                return "No prompt to refine. Please optimize a prompt first."
            
            # Determine which rules to apply based on feedback
            rules_to_apply = []
            if "structure" in feedback.lower():
                rules_to_apply.append("improve_structure")
            if "steps" in feedback.lower():
                rules_to_apply.append("add_steps_section")
            if "clarity" in feedback.lower():
                rules_to_apply.append("add_clarity_guidelines")
            if "context" in feedback.lower():
                rules_to_apply.append("add_context_preservation")
            if "redundant" in feedback.lower() or "repetitive" in feedback.lower():
                rules_to_apply.append("remove_redundancy")
            
            # If no specific rules identified, re-optimize the entire prompt
            if not rules_to_apply:
                return self.optimize_prompt(latest.original_prompt)
            
            # Apply specified rules to the latest optimized prompt
            optimized = latest.optimized_prompt
            applied_rules = []
            
            for rule_name in rules_to_apply:
                for rule in self.optimization_engine.rules:
                    if rule.name == rule_name:
                        before = optimized
                        optimized = rule.apply(optimized)
                        if optimized != before:
                            applied_rules.append(rule.name)
            
            # Create prompt data for history
            version = latest.version + 1
            
            prompt_data = PromptData(
                original_prompt=latest.original_prompt,
                optimized_prompt=optimized,
                version=version,
                stage=OptimizationStage.REFINEMENT,
                rules_applied=applied_rules,
                user_feedback=[feedback]
            )
            
            # Add to history
            self.optimization_history.add_version(prompt_data)
            
            # Log output for server monitoring
            console.print(f"[bold green]Refined prompt v{version}[/bold green]")
            console.print(f"Applied rules: {', '.join(applied_rules)}")
            console.print(f"User feedback: {feedback}")
            
            # Return only the optimized prompt
            return optimized
            
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")
            return f"Error refining prompt: {str(e)}"
    
    def get_optimization_history(self) -> str:
        """Get the full optimization history."""
        if not self.optimization_history.history:
            return json.dumps({"history": "No optimization history available"})
        
        history = []
        for prompt_data in self.optimization_history.history:
            history.append({
                "version": prompt_data.version,
                "stage": prompt_data.stage.name,
                "rulesApplied": prompt_data.rules_applied,
                "timestamp": prompt_data.created_at.isoformat()
            })
        
        return json.dumps({"history": history}, indent=2)
    
    def clear_history(self) -> str:
        """Clear the optimization history."""
        self.optimization_history.clear()
        return json.dumps({"status": "success", "message": "Optimization history cleared"})

def create_server() -> FastMCP:
    """Create and configure the MCP server."""
    mcp = FastMCP("prompt-optimization")
    optimization_server = PromptEngineeringServer()
    
    @mcp.tool()
    async def optimize_prompt(prompt: str) -> str:
        """
        Automatically optimize a prompt based on best practices.
        
        Args:
            prompt: The original prompt to optimize
            
        Returns:
            Optimized prompt
        """
        return optimization_server.optimize_prompt(prompt)

    @mcp.tool()
    async def refine_prompt(feedback: str) -> str:
        """
        Refine the current prompt based on user feedback.
        
        Args:
            feedback: User feedback for further refinement
            
        Returns:
            Refined prompt
        """
        return optimization_server.refine_prompt(feedback)

    @mcp.tool()
    async def get_optimization_history() -> str:
        """
        Get the full optimization history.
        
        Returns:
            JSON string containing the optimization history
        """
        return optimization_server.get_optimization_history()

    @mcp.tool()
    async def clear_optimization_history() -> str:
        """
        Clear the optimization history.
        
        Returns:
            Confirmation message
        """
        return optimization_server.clear_history()

    return mcp

def main():
    """Main entry point for the prompt optimization server."""
    server = create_server()
    console.print("[bold green]Prompt Optimization Server Starting...[/bold green]")
    return server.run()

if __name__ == "__main__":
    try:
        server = create_server()
        console.print("[bold green]Prompt Optimization Server[/bold green]")
        console.print("Version: 1.0.0")
        console.print("Available stages:", ", ".join(stage.value for stage in OptimizationStage))
        server.run()
    except Exception as e:
        console.print(f"[bold red]Fatal Error:[/bold red] {str(e)}")
        raise