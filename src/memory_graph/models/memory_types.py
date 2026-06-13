"""Enumeration types for memory nodes and edges in the cognitive graph."""

from enum import Enum  # Standard library enum for type-safe constants


class NodeType(str, Enum):
    """Classification of memory node types mirroring human cognitive categories."""

    THOUGHT = "thought"  # A general thought or observation held by the person
    BELIEF = "belief"  # A held belief or conviction about the world
    AVERSION = "aversion"  # Something the person dislikes or actively avoids
    PREFERENCE = "preference"  # Something the person prefers or gravitates toward
    EMOTION = "emotion"  # An emotional state or recurring feeling
    GOAL = "goal"  # A desired outcome or aspiration they are working toward
    EXPERIENCE = "experience"  # A past event or lived experience
    FACT = "fact"  # A factual piece of information about the person
    HABIT = "habit"  # A recurring behavioral pattern
    VALUE = "value"  # A core value or guiding principle
    RELATIONSHIP = "relationship"  # Information about a relationship with another person
    SKILL = "skill"  # A known skill or area of competency
    CUSTOM = "custom"  # User-defined type requiring custom_type_name field


class EdgeType(str, Enum):
    """Classification of relationship types between memory nodes."""

    CONTRADICTS = "contradicts"  # Memory A directly contradicts memory B
    SUPPORTS = "supports"  # Memory A provides evidence for memory B
    CAUSED_BY = "caused_by"  # Memory A was caused by or triggered by memory B
    LEADS_TO = "leads_to"  # Memory A leads to or results in memory B
    RELATED_TO = "related_to"  # General semantic association between memories
    EVOLVED_FROM = "evolved_from"  # Memory A is an updated version of memory B
    CONFLICTS_WITH = "conflicts_with"  # Tension exists between memory A and B
    DEPENDS_ON = "depends_on"  # Memory A requires memory B to be true
    TEMPORAL_BEFORE = "temporal_before"  # Memory A occurred before memory B in time
    TEMPORAL_AFTER = "temporal_after"  # Memory A occurred after memory B in time
    PART_OF = "part_of"  # Memory A is a component or subset of memory B
    GENERALIZES = "generalizes"  # Memory A is a broader generalization of memory B
    SPECIFIES = "specifies"  # Memory A is a specific instance of memory B


class EmotionValence(str, Enum):
    """Emotional polarity classification for memory nodes."""

    POSITIVE = "positive"  # Associated with positive emotional tone
    NEGATIVE = "negative"  # Associated with negative emotional tone
    NEUTRAL = "neutral"  # No strong emotional association
    MIXED = "mixed"  # Contains both positive and negative elements
