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
    
    @classmethod
    def from_string(cls, value: str) -> 'OptimizationStage':
        """Convert string to OptimizationStage with better error handling."""
        try:
            # Try direct conversion first
            return cls(value)
        except ValueError:
            # Try case-insensitive match
            upper_value = value.upper()
            for stage in cls:
                if stage.name.upper() == upper_value:
                    return stage
            # Try matching the value part
            for stage in cls:
                if stage.value.upper() == upper_value:
                    return stage
            raise ValueError(f"Invalid stage: {value}. Valid stages are: {[stage.value for stage in cls]}")

@dataclass
class PromptRule:
    """Rule for prompt optimization with implementation."""
    name: str
    description: str
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
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> None:
        """Validate prompt data consistency."""
        if not self.original_prompt:
            raise ValueError("Original prompt cannot be empty")
        if self.version < 1:
            raise ValueError("Version must be at least 1")

@dataclass
class OptimizationHistory:
    history: List[PromptData] = field(default_factory=list)
    conversation: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_version(self, prompt_data: PromptData) -> None:
        """Add a new version to the history."""
        self.history.append(prompt_data)
    
    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation history."""
        self.conversation.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
    
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
        self.conversation.clear()

class PromptOptimizationEngine:
    """Engine for automatically optimizing prompts based on rules."""
    
    def __init__(self):
        # Define optimization rules - easily editable
        self.rules = [
            PromptRule(
                name="Add Steps Section",
                description="Add a 'steps' section to guide completion",
                apply=self._add_steps_section
            ),
            PromptRule(
                name="Improve Structure",
                description="Structure the prompt with clear sections",
                apply=self._improve_structure
            ),
            PromptRule(
                name="Add Context Preservation",
                description="Ensure important context is preserved",
                apply=self._add_context_preservation
            ),
            PromptRule(
                name="Add Clarity Guidelines",
                description="Add guidelines for clear outputs",
                apply=self._add_clarity_guidelines
            ),
            PromptRule(
                name="Remove Redundancy",
                description="Remove redundant or repetitive elements",
                apply=self._remove_redundancy
            )
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
    
    def optimize_prompt(self, prompt: str, specific_rules: Optional[List[str]] = None) -> Dict[str, Any]:
        """Apply optimization rules to improve a prompt."""
        optimized = prompt
        applied_rules = []
        
        # Apply either specific rules or all rules
        rules_to_apply = self.rules
        if specific_rules:
            rules_to_apply = [rule for rule in self.rules if rule.name in specific_rules]
        
        for rule in rules_to_apply:
            before = optimized
            optimized = rule.apply(optimized)
            if optimized != before:
                applied_rules.append(rule.name)
        
        return {
            "optimized_prompt": optimized,
            "rules_applied": applied_rules
        }

class PromptEngineeringServer:
    def __init__(self):
        self.optimization_history = OptimizationHistory()
        self.optimization_engine = PromptOptimizationEngine()
        
        # Hardcoded configuration - easily editable
        self.default_template = """Your task for today is to perform prompt engineering on a prompt. The next message will be divided into XML tags.
* <rules>: Describes the rules you must follow to improve the prompt
* <prompt>: The actual prompt
* <notes>: Any comments that come to mind
* <tools>: The tools you will use for redefining the prompt.

<rules>
* The prompt must be redefined while preserving all information
* The final prompt must include a "steps" section that describes to the LLM the necessary steps to complete the task optimally
* You must write a final artifact with the refined prompt
</rules>

<notes>
* The prompt is for a DeepSearch
* Ask questions to focus the prompt more clearly. Use Brave Search to help refine the focus.
</notes>

<tools>
* Use Brave Search to verify assumptions
* Use Sequential Thinking to verify logical flow
</tools>

<prompt>
{original_prompt}
</prompt>
"""
    
    def _format_prompt(self, prompt_data: PromptData) -> Panel:
        """Format a prompt for display."""
        stage_colors = {
            OptimizationStage.INITIAL_ANALYSIS: "bold red",
            OptimizationStage.RULES_APPLICATION: "bold yellow",
            OptimizationStage.STRUCTURING: "bold blue",
            OptimizationStage.VERIFICATION: "bold green",
            OptimizationStage.REFINEMENT: "bold magenta",
            OptimizationStage.FINAL: "bold cyan"
        }
        
        title = f"Prompt Engineering v{prompt_data.version} - {prompt_data.stage.value}"
            
        content = Group(
            Text(f"Original Prompt:", style="bold"),
            Text(prompt_data.original_prompt),
            Text(f"\nEngineered Prompt:", style="bold"),
            Text(prompt_data.optimized_prompt),
            Text(f"\nRules Applied: {', '.join(prompt_data.rules_applied)}" if prompt_data.rules_applied else "")
        )
        
        return Panel(
            content,
            title=title,
            border_style=stage_colors.get(prompt_data.stage, "white")
        )
    
    def optimize_prompt(self, prompt: str, specific_rules: Optional[List[str]] = None) -> dict:
        """Optimize a prompt based on rules."""
        try:
            # Get optimization result
            optimization_result = self.optimization_engine.optimize_prompt(prompt, specific_rules)
            
            # Create prompt data
            latest = self.optimization_history.get_latest()
            version = (latest.version + 1) if latest else 1
            
            # Create final prompt data with template
            final_prompt = self.default_template.format(
                original_prompt=optimization_result["optimized_prompt"]
            )
            
            prompt_data = PromptData(
                original_prompt=prompt,
                optimized_prompt=final_prompt,
                version=version,
                stage=OptimizationStage.RULES_APPLICATION,
                rules_applied=optimization_result["rules_applied"]
            )
            
            # Add to history
            self.optimization_history.add_version(prompt_data)
            self.optimization_history.add_message("system", "Prompt optimization applied")
            
            # Display formatted prompt
            console.print(self._format_prompt(prompt_data))
            
            # Return response
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps({
                        "promptOptimization": {
                            "version": prompt_data.version,
                            "originalPrompt": prompt_data.original_prompt,
                            "optimizedPrompt": prompt_data.optimized_prompt,
                            "rulesApplied": prompt_data.rules_applied,
                            "stage": prompt_data.stage.value,
                            "timestamp": prompt_data.created_at.isoformat()
                        }
                    }, indent=2)
                }]
            }
            
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps({
                        "error": str(e),
                        "status": "failed",
                        "errorType": type(e).__name__,
                        "timestamp": datetime.now().isoformat()
                    }, indent=2)
                }],
                "isError": True
            }
    
    def refine_prompt(self, feedback: str) -> dict:
        """Refine a prompt based on user feedback."""
        try:
            latest = self.optimization_history.get_latest()
            if not latest:
                return {
                    "content": [{
                        "type": "text",
                        "text": json.dumps({
                            "error": "No prompt to refine. Please optimize a prompt first.",
                            "status": "failed"
                        }, indent=2)
                    }],
                    "isError": True
                }
            
            # Create a new version based on feedback
            version = latest.version + 1
            
            # Analyze feedback to determine what to improve
            if "more structure" in feedback.lower():
                rule_to_apply = "Improve Structure"
            elif "steps" in feedback.lower():
                rule_to_apply = "Add Steps Section"
            elif "clarity" in feedback.lower():
                rule_to_apply = "Add Clarity Guidelines"
            elif "context" in feedback.lower():
                rule_to_apply = "Add Context Preservation"
            elif "redundant" in feedback.lower() or "repetitive" in feedback.lower():
                rule_to_apply = "Remove Redundancy"
            else:
                # Apply all rules if feedback doesn't match specific rules
                rule_to_apply = None
            
            # Apply optimization based on feedback
            optimization_result = self.optimization_engine.optimize_prompt(
                latest.optimized_prompt, 
                [rule_to_apply] if rule_to_apply else None
            )
            
            # Create final prompt data with template
            final_prompt = self.default_template.format(
                original_prompt=optimization_result["optimized_prompt"]
            )
            
            # Create new prompt data
            prompt_data = PromptData(
                original_prompt=latest.original_prompt,
                optimized_prompt=final_prompt,
                version=version,
                stage=OptimizationStage.REFINEMENT,
                rules_applied=optimization_result["rules_applied"],
                user_feedback=[feedback]
            )
            
            # Add to history
            self.optimization_history.add_version(prompt_data)
            self.optimization_history.add_message("user", feedback)
            self.optimization_history.add_message("system", "Prompt refinement applied")
            
            # Display formatted prompt
            console.print(self._format_prompt(prompt_data))
            
            # Return response
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps({
                        "promptRefinement": {
                            "version": prompt_data.version,
                            "previousVersion": latest.version,
                            "optimizedPrompt": prompt_data.optimized_prompt,
                            "rulesApplied": prompt_data.rules_applied,
                            "userFeedback": feedback,
                            "stage": prompt_data.stage.value,
                            "timestamp": prompt_data.created_at.isoformat()
                        }
                    }, indent=2)
                }]
            }
            
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps({
                        "error": str(e),
                        "status": "failed",
                        "errorType": type(e).__name__,
                        "timestamp": datetime.now().isoformat()
                    }, indent=2)
                }],
                "isError": True
            }
    
    def finalize_prompt(self) -> dict:
        """Finalize the current prompt optimization."""
        try:
            latest = self.optimization_history.get_latest()
            if not latest:
                return {
                    "content": [{
                        "type": "text",
                        "text": json.dumps({
                            "error": "No prompt to finalize. Please optimize a prompt first.",
                            "status": "failed"
                        }, indent=2)
                    }],
                    "isError": True
                }
            
            # Create a final version
            prompt_data = PromptData(
                original_prompt=latest.original_prompt,
                optimized_prompt=latest.optimized_prompt,
                version=latest.version + 1,
                stage=OptimizationStage.FINAL,
                rules_applied=latest.rules_applied,
                user_feedback=latest.user_feedback
            )
            
            # Add to history
            self.optimization_history.add_version(prompt_data)
            self.optimization_history.add_message("system", "Prompt finalized")
            
            # Display formatted prompt
            console.print(self._format_prompt(prompt_data))
            
            # Return response
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps({
                        "promptFinalization": {
                            "version": prompt_data.version,
                            "originalPrompt": prompt_data.original_prompt,
                            "finalPrompt": prompt_data.optimized_prompt,
                            "rulesApplied": prompt_data.rules_applied,
                            "stage": prompt_data.stage.value,
                            "timestamp": prompt_data.created_at.isoformat()
                        }
                    }, indent=2)
                }]
            }
            
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps({
                        "error": str(e),
                        "status": "failed",
                        "errorType": type(e).__name__,
                        "timestamp": datetime.now().isoformat()
                    }, indent=2)
                }],
                "isError": True
            }
    
    def get_optimization_history(self) -> str:
        """Get the full optimization history."""
        if not self.optimization_history.history:
            return json.dumps({"history": "No optimization history available"})
        
        history = []
        for prompt_data in self.optimization_history.history:
            history.append({
                "version": prompt_data.version,
                "stage": prompt_data.stage.value,
                "rulesApplied": prompt_data.rules_applied,
                "timestamp": prompt_data.created_at.isoformat()
            })
        
        return json.dumps({
            "history": history,
            "conversation": self.optimization_history.conversation
        }, indent=2)
    
    def get_optimized_prompt(self, version: Optional[int] = None) -> str:
        """Get a specific optimized prompt or the latest one."""
        if version:
            prompt_data = self.optimization_history.get_version(version)
            if not prompt_data:
                return json.dumps({"error": f"Version {version} not found"})
        else:
            prompt_data = self.optimization_history.get_latest()
            if not prompt_data:
                return json.dumps({"error": "No optimization history available"})
        
        return json.dumps({
            "version": prompt_data.version,
            "stage": prompt_data.stage.value,
            "optimizedPrompt": prompt_data.optimized_prompt,
            "rulesApplied": prompt_data.rules_applied,
            "timestamp": prompt_data.created_at.isoformat()
        }, indent=2)
    
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
            JSON string containing the optimized prompt
        """
        result = optimization_server.optimize_prompt(prompt)
        return result["content"][0]["text"]

    @mcp.tool()
    async def refine_prompt(feedback: str) -> str:
        """
        Refine the current prompt based on user feedback.
        
        Args:
            feedback: User feedback for further refinement
            
        Returns:
            JSON string containing the refined prompt
        """
        result = optimization_server.refine_prompt(feedback)
        return result["content"][0]["text"]

    @mcp.tool()
    async def finalize_prompt() -> str:
        """
        Finalize the current prompt optimization.
        
        Returns:
            JSON string containing the final optimized prompt
        """
        result = optimization_server.finalize_prompt()
        return result["content"][0]["text"]

    @mcp.tool()
    async def get_optimization_history() -> str:
        """
        Get the full optimization history including conversation.
        
        Returns:
            JSON string containing the optimization history
        """
        return optimization_server.get_optimization_history()

    @mcp.tool()
    async def get_optimized_prompt(version: Optional[int] = None) -> str:
        """
        Get a specific optimized prompt or the latest one.
        
        Args:
            version: Specific version to retrieve (optional)
            
        Returns:
            JSON string containing the optimized prompt
        """
        return optimization_server.get_optimized_prompt(version)

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