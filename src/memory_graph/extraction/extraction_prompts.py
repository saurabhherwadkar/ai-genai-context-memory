"""System prompts for the memory extraction LLM pipeline."""

# System prompt instructing the LLM to extract structured memories from conversations
EXTRACTION_SYSTEM_PROMPT = """You are a cognitive memory extraction system. Your task is to analyze conversations and extract discrete memory items that reveal information about the user's thoughts, beliefs, preferences, aversions, emotions, goals, experiences, facts, habits, values, relationships, and skills.

For each memory you extract, you MUST output a JSON object with these fields:
- "content": The full description of the memory (1-3 sentences)
- "summary": A one-line summary (under 100 characters)
- "node_type": One of: thought, belief, aversion, preference, emotion, goal, experience, fact, habit, value, relationship, skill, custom
- "confidence": How confident you are this is correct (0.0 to 1.0)
- "emotional_valence": One of: positive, negative, neutral, mixed
- "intensity": How strong/important this memory is (0.0 to 1.0)
- "tags": A list of 1-5 relevant category tags
- "suggested_edges": A list of objects with "target_content_hint" (brief text of what it relates to) and "edge_type" (one of: contradicts, supports, caused_by, leads_to, related_to, evolved_from, conflicts_with, depends_on)

Rules:
1. Only extract memories that are clearly stated or strongly implied - do NOT speculate
2. Focus on ENDURING information (preferences, beliefs, goals) over transient states
3. Each memory should be a single, atomic piece of information
4. Assign higher confidence to explicitly stated information vs. implied
5. Extract a maximum of 5 memories per conversation
6. Do NOT extract information about the AI assistant, only about the user
7. Do NOT extract generic knowledge or facts unrelated to the user

Output format: Return ONLY a JSON object with a single key "extracted_memories" containing a list of memory objects. No other text."""

# Template for the extraction user prompt, filled with conversation content
EXTRACTION_USER_PROMPT_TEMPLATE = """Analyze the following conversation and extract memory items about the user.

CONVERSATION:
{conversation_text}

Extract up to {max_memories} memory items. Return ONLY valid JSON."""


def build_extraction_prompt(
    conversation_text: str,
    max_memories: int = 5,
) -> tuple[str, str]:
    """Build the system and user prompts for memory extraction.

    Returns a tuple of (system_prompt, user_prompt) ready for LLM submission.
    """
    # Format the user prompt with conversation content
    user_prompt = EXTRACTION_USER_PROMPT_TEMPLATE.format(  # Fill template placeholders
        conversation_text=conversation_text,  # Insert conversation
        max_memories=max_memories,  # Insert memory limit
    )

    return (EXTRACTION_SYSTEM_PROMPT, user_prompt)  # Return both prompts


def format_conversation_for_extraction(
    messages: list[dict],
    response_content: str,
) -> str:
    """Format a conversation exchange into text for the extraction prompt.

    Converts message objects into a readable conversation transcript.
    """
    lines: list[str] = []  # Accumulate transcript lines

    for message in messages:  # Process each message in the conversation
        role = message.get("role", "unknown")  # Get the speaker role
        content = message.get("content", "")  # Get the message text

        if isinstance(content, list):  # Multimodal content (list of blocks)
            text_parts = [  # Extract text from content blocks
                item.get("text", "") for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            ]
            content = " ".join(text_parts)  # Join text parts

        if role == "system":  # Skip system messages (not user info)
            continue  # Don't include system prompts in extraction

        # Format the message with role label
        role_label = "User" if role == "user" else "Assistant"  # Human-readable label
        lines.append(f"{role_label}: {content}")  # Add formatted line

    # Add the final assistant response
    lines.append(f"Assistant: {response_content}")  # Append response

    return "\n".join(lines)  # Return complete transcript
