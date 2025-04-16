# Prompt Name: data_analysis_prompt

## Description
Data analysis and interpretation template for insights and recommendations.

## Tags
task: analysis
type: data_analysis

## Template
Analyze the following {{ data_type }} data about {{ subject }} and provide insights:

```
{{ data }}
```

Your analysis should include:

1. Summary of the key metrics and patterns
   - What are the main trends visible in the data?
   - Are there any outliers or anomalies?

2. In-depth analysis of {{ specific_aspect }}
   - What factors might be influencing these results?
   - How does this compare to industry benchmarks or expectations?

3. Correlations and relationships
   - What variables appear to be related?
   - Is there evidence of causation vs. correlation?

4. Actionable insights
   - What decisions could be made based on this data?
   - What opportunities or risks does this data reveal?

5. Recommendations for further analysis
   - What additional data would be valuable?
   - What questions remain unanswered?

Format your response with clearly labeled sections, and where appropriate, suggest how this data might be visualized for maximum impact.