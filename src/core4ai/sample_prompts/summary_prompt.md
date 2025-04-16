# Prompt Name: summary_prompt

## Description
Create concise, accurate summaries of long-form content.

## Tags
task: summarization
type: summary

## Template
Create a {{ length }} summary of the following {{ content_type }}:

"""
{{ content }}
"""

Your summary should:
- Capture the main ideas and key points
- Maintain the original meaning and intent
- Eliminate unnecessary details and examples
- Follow a logical structure with clear transitions
- Be objective and balanced in presenting all viewpoints
- Use simple, clear language while preserving essential terminology
- End with the most important conclusion or implication

Format the summary to be easily scannable with appropriate paragraphs and bullet points where helpful.