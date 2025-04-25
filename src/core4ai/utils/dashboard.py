"""
Enhanced dashboard generator for Core4AI analytics with fixed success rate calculation.

This module provides functionality to generate modern, responsive HTML dashboards
from Core4AI analytics data with a focus on user experience and visual appeal.
"""
import os
import json
import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

def calculate_success_rate(analytics_data, usage_data):
    """
    Calculate the success rate from analytics data correctly.
    
    Args:
        analytics_data: The prompt analytics data
        usage_data: The usage statistics data
        
    Returns:
        Correctly calculated success rate
    """
    # First try to get success rate from provider stats which is more reliable
    provider_stats = usage_data.get('provider_stats', [])
    if provider_stats:
        # Calculate weighted average if we have multiple providers
        total_count = 0
        total_success_weighted = 0
        
        for provider in provider_stats:
            count = provider.get('count', 0)
            success_rate = provider.get('success_rate', 0)
            
            if count > 0:
                total_count += count
                total_success_weighted += count * success_rate
        
        if total_count > 0:
            # Return the weighted average success rate
            return round(total_success_weighted / total_count, 2)
    
    # Fallback to prompt metrics if provider stats don't have success rates
    prompt_metrics = analytics_data.get('metrics', [])
    if prompt_metrics:
        success_rates = []
        for metric in prompt_metrics:
            rate = metric.get('success_rate')
            if rate is not None:
                success_rates.append(rate)
        
        if success_rates:
            return round(sum(success_rates) / len(success_rates), 2)
    
    # If we can't find success rates, check for successful flags in usage data
    usage_data_list = analytics_data.get('usage_data', [])
    if usage_data_list:
        successful_count = sum(1 for item in usage_data_list if item.get('successful', False))
        total_items = len(usage_data_list)
        
        if total_items > 0:
            return round((successful_count / total_items) * 100, 2)
    
    # Default to 100 if we can't calculate
    return 100.0

def generate_dashboard(
    analytics_data: Dict[str, Any],
    usage_data: Dict[str, Any],
    output_dir: Optional[str] = None,
    filename: Optional[str] = None
) -> str:
    """
    Generate an enhanced HTML dashboard for Core4AI analytics.
    
    Args:
        analytics_data: The prompt analytics data
        usage_data: The usage statistics data
        output_dir: Directory to save the dashboard (default: current directory)
        filename: Filename for the dashboard (default: coreai_stats_timestamp.html)
        
    Returns:
        Path to the generated dashboard file
    """
    # Default to current directory if not specified
    if output_dir is None:
        output_dir = os.getcwd()
    else:
        # Ensure the directory exists
        os.makedirs(output_dir, exist_ok=True)
    
    # Generate default filename with timestamp if not provided
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"coreai_stats_{timestamp}.html"
    
    # Ensure filename has .html extension
    if not filename.endswith('.html'):
        filename += '.html'
    
    # Full path to the dashboard file
    dashboard_path = os.path.join(output_dir, filename)
    
    # Extract data for charts
    usage_by_prompt = usage_data.get('usage_by_prompt', [])
    usage_by_day = usage_data.get('usage_by_day', [])
    provider_stats = usage_data.get('provider_stats', [])
    content_stats = usage_data.get('content_stats', [])
    total_count = usage_data.get('total_count', 0)
    
    # Generate dates and counts for the time series chart
    dates = [item.get('date', '') for item in usage_by_day]
    counts = [item.get('count', 0) for item in usage_by_day]
    
    # Generate data for prompt usage chart (limit to top 10)
    prompt_names = [item.get('prompt_name', '').replace('_prompt', '') for item in usage_by_prompt[:10]]
    prompt_counts = [item.get('count', 0) for item in usage_by_prompt[:10]]
    
    # Calculate prompt usage percentages for pie chart
    total_prompt_usage = sum(prompt_counts)
    prompt_percentages = []
    for count in prompt_counts:
        if total_prompt_usage > 0:
            percentage = round((count / total_prompt_usage) * 100, 1)
        else:
            percentage = 0
        prompt_percentages.append(percentage)
    
    # Generate provider data
    provider_names = []
    provider_counts = []
    for item in provider_stats:
        provider = item.get('provider', 'unknown')
        model = item.get('model', 'default')
        if provider and model:
            provider_names.append(f"{provider} {model}")
            provider_counts.append(item.get('count', 0))
    
    # Generate content type data
    content_types = []
    content_counts = []
    for item in content_stats:
        content_type = item.get('content_type', '')
        if content_type:
            content_types.append(content_type)
            content_counts.append(item.get('count', 0))
    
    # Calculate success rates and averages using our fixed function
    success_rate = calculate_success_rate(analytics_data, usage_data)
    
    # Calculate average confidence score
    avg_confidence = 0
    prompt_metrics = analytics_data.get('metrics', [])
    
    if prompt_metrics:
        confidence_scores = [metric.get('avg_confidence', 0) for metric in prompt_metrics if metric.get('avg_confidence') is not None]
        if confidence_scores:
            avg_confidence = round(sum(confidence_scores) / len(confidence_scores), 2)
    
    # Get usage timeline data and format it for the chart
    timeline_data = []
    for day_data in usage_by_day:
        date = day_data.get('date', '')
        count = day_data.get('count', 0)
        if date:
            timeline_data.append({"date": date, "count": count})
    
    # Format data for provider performance chart
    provider_performance = []
    for i, provider in enumerate(provider_names):
        provider_performance.append({
            "provider": provider,
            "count": provider_counts[i] if i < len(provider_counts) else 0
        })
    
    # Format data for prompt type distribution chart
    prompt_distribution = []
    for i, prompt in enumerate(prompt_names):
        if i < len(prompt_percentages):
            prompt_distribution.append({
                "name": prompt,
                "percentage": prompt_percentages[i],
                "count": prompt_counts[i] if i < len(prompt_counts) else 0
            })
    
    # Generate key insights
    insights = []
    if prompt_names and prompt_percentages:
        most_used = prompt_names[0]
        most_used_pct = prompt_percentages[0]
        insights.append(f"{most_used.capitalize()} prompts are the most frequently used ({most_used_pct}%)")
    
    insights.append(f"Providers have {success_rate}% success rates")
    
    if provider_names and provider_counts:
        popular_models = ", ".join([p for p, c in zip(provider_names, provider_counts) if c > 0][:2])
        insights.append(f"{popular_models} are the most popular models")
    
    insights.append(f"Average confidence score across all prompts: {avg_confidence}%")
    
    # Create the HTML content using a modern template
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Core4AI Analytics Dashboard</title>
    <!-- Include Tailwind CSS via CDN -->
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- Include Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <!-- Include date-fns for better date handling -->
    <script src="https://cdn.jsdelivr.net/npm/date-fns@2.30.0/index.min.js"></script>
    <!-- Add some custom styles -->
    <style>
        .chart-container {{
            position: relative;
            height: 300px;
            width: 100%;
        }}
        
        .insights-list li {{
            margin-bottom: 0.75rem;
            display: flex;
            align-items: flex-start;
        }}
        
        .insights-list li::before {{
            content: "•";
            color: #3b82f6;
            font-weight: bold;
            font-size: 1.25rem;
            margin-right: 0.5rem;
            line-height: 1;
        }}
        
        .card {{
            transition: all 0.3s ease;
        }}
        
        .card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
        }}
        
        @media (max-width: 768px) {{
            .chart-container {{
                height: 220px;
            }}
        }}
    </style>
</head>
<body class="bg-gray-50 font-sans">
    <div class="container mx-auto px-4 py-8 max-w-7xl">
        <!-- Header -->
        <div class="relative overflow-hidden bg-gradient-to-r from-blue-600 to-indigo-700 rounded-xl shadow-lg mb-8">
            <div class="absolute inset-0 opacity-10">
                <svg class="h-full w-full" viewBox="0 0 400 400" xmlns="http://www.w3.org/2000/svg">
                    <defs>
                        <pattern id="small-grid" width="10" height="10" patternUnits="userSpaceOnUse">
                            <path d="M 10 0 L 0 0 0 10" fill="none" stroke="white" stroke-width="0.5"></path>
                        </pattern>
                        <pattern id="grid" width="100" height="100" patternUnits="userSpaceOnUse">
                            <rect width="100" height="100" fill="url(#small-grid)"></rect>
                            <path d="M 100 0 L 0 0 0 100" fill="none" stroke="white" stroke-width="1"></path>
                        </pattern>
                    </defs>
                    <rect width="100%" height="100%" fill="url(#grid)"></rect>
                </svg>
            </div>
            <div class="relative px-8 py-10">
                <div class="flex flex-col md:flex-row justify-between items-start md:items-center">
                    <div>
                        <h1 class="text-3xl font-bold text-white">Core4AI Analytics Dashboard</h1>
                        <p class="text-blue-100 mt-2">
                            Generated on {datetime.datetime.now().strftime("%B %d, %Y at %H:%M:%S")}
                        </p>
                    </div>
                    <div class="mt-4 md:mt-0 bg-white bg-opacity-20 backdrop-filter backdrop-blur rounded-lg px-4 py-2 text-white">
                        <div class="text-sm">Interactions</div>
                        <div class="text-3xl font-bold">{total_count}</div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Stats Cards -->
        <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
            <div class="card bg-white rounded-xl shadow p-6">
                <div class="flex items-center">
                    <div class="rounded-full bg-blue-100 p-3 mr-4">
                        <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                        </svg>
                    </div>
                    <div>
                        <div class="text-sm font-medium text-gray-500">Prompt Types</div>
                        <div class="text-2xl font-bold text-gray-900">{len(usage_by_prompt)}</div>
                    </div>
                </div>
                <div class="mt-4 flex justify-between items-center">
                    <div class="text-sm text-gray-500">Usage analytics</div>
                    <div class="text-sm font-medium text-green-600">
                        <span class="bg-green-100 px-2 py-1 rounded">Active</span>
                    </div>
                </div>
            </div>
            
            <div class="card bg-white rounded-xl shadow p-6">
                <div class="flex items-center">
                    <div class="rounded-full bg-indigo-100 p-3 mr-4">
                        <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6 text-indigo-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                        </svg>
                    </div>
                    <div>
                        <div class="text-sm font-medium text-gray-500">Success Rate</div>
                        <div class="text-2xl font-bold text-gray-900">{success_rate}%</div>
                    </div>
                </div>
                <div class="mt-4 flex justify-between items-center">
                    <div class="text-sm text-gray-500">Across all interactions</div>
                    <div class="text-sm font-medium text-blue-600">
                        <span class="bg-blue-100 px-2 py-1 rounded">Excellent</span>
                    </div>
                </div>
            </div>
            
            <div class="card bg-white rounded-xl shadow p-6">
                <div class="flex items-center">
                    <div class="rounded-full bg-purple-100 p-3 mr-4">
                        <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6 text-purple-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
                        </svg>
                    </div>
                    <div>
                        <div class="text-sm font-medium text-gray-500">Confidence Score</div>
                        <div class="text-2xl font-bold text-gray-900">{avg_confidence}%</div>
                    </div>
                </div>
                <div class="mt-4 flex justify-between items-center">
                    <div class="text-sm text-gray-500">Average across prompts</div>
                    <div class="text-sm font-medium text-purple-600">
                        <span class="bg-purple-100 px-2 py-1 rounded">High</span>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Usage Timeline Chart -->
        <div class="card bg-white rounded-xl shadow mb-8">
            <div class="p-6">
                <h2 class="text-xl font-bold text-gray-900 mb-4">Usage Timeline</h2>
                <div class="chart-container">
                    <canvas id="timelineChart"></canvas>
                </div>
            </div>
        </div>
        
        <!-- Two Column Charts -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
            <!-- Prompt Distribution -->
            <div class="card bg-white rounded-xl shadow">
                <div class="p-6">
                    <h2 class="text-xl font-bold text-gray-900 mb-4">Prompt Distribution</h2>
                    <div class="chart-container">
                        <canvas id="promptDistribution"></canvas>
                    </div>
                </div>
            </div>
            
            <!-- Provider Performance -->
            <div class="card bg-white rounded-xl shadow">
                <div class="p-6">
                    <h2 class="text-xl font-bold text-gray-900 mb-4">Provider Performance</h2>
                    <div class="chart-container">
                        <canvas id="providerPerformance"></canvas>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Key Insights -->
        <div class="card bg-white rounded-xl shadow mb-8">
            <div class="p-6">
                <h2 class="text-xl font-bold text-gray-900 mb-4">Key Insights</h2>
                <ul class="insights-list text-gray-700">
                    {' '.join(f'<li>{insight}</li>' for insight in insights)}
                </ul>
            </div>
        </div>
        
        <!-- Footer -->
        <div class="text-center text-gray-500 text-sm mt-12 mb-6">
            <p>Powered by Core4AI Analytics - Data from past {usage_data.get('time_range', 30)} days ({total_count} total interactions)</p>
        </div>
    </div>

    <!-- Chart.js Initialization -->
    <script>
        // Helper function to create gradient
        function createGradient(ctx, colors) {{
            const gradient = ctx.createLinearGradient(0, 0, 0, 400);
            gradient.addColorStop(0, colors[0]);
            gradient.addColorStop(1, colors[1]);
            return gradient;
        }}
        
        // Timeline Chart
        const timelineCtx = document.getElementById('timelineChart').getContext('2d');
        const timelineGradient = createGradient(timelineCtx, ['rgba(59, 130, 246, 0.4)', 'rgba(59, 130, 246, 0.0)']);
        
        new Chart(timelineCtx, {{
            type: 'line',
            data: {{
                labels: {json.dumps(dates)},
                datasets: [{{
                    label: 'Interactions',
                    data: {json.dumps(counts)},
                    borderColor: '#3b82f6',
                    backgroundColor: timelineGradient,
                    borderWidth: 3,
                    pointBackgroundColor: '#fff',
                    pointBorderColor: '#3b82f6',
                    pointRadius: 4,
                    pointHoverRadius: 6,
                    tension: 0.3,
                    fill: true
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        display: false
                    }},
                    tooltip: {{
                        backgroundColor: 'rgba(17, 24, 39, 0.9)',
                        titleColor: '#f3f4f6',
                        bodyColor: '#f3f4f6',
                        padding: 12,
                        cornerRadius: 8,
                        displayColors: false,
                        callbacks: {{
                            title: function(context) {{
                                return context[0].label;
                            }},
                            label: function(context) {{
                                return context.raw + ' interactions';
                            }}
                        }}
                    }}
                }},
                scales: {{
                    x: {{
                        grid: {{
                            display: false
                        }},
                        ticks: {{
                            font: {{
                                size: 11
                            }},
                            color: '#6b7280'
                        }}
                    }},
                    y: {{
                        beginAtZero: true,
                        grid: {{
                            color: 'rgba(243, 244, 246, 1)',
                            drawBorder: false
                        }},
                        ticks: {{
                            precision: 0,
                            font: {{
                                size: 11
                            }},
                            color: '#6b7280',
                            padding: 8
                        }}
                    }}
                }},
                interaction: {{
                    intersect: false,
                    mode: 'index'
                }}
            }}
        }});
        
        // Prompt Distribution Chart
        const promptCtx = document.getElementById('promptDistribution').getContext('2d');
        new Chart(promptCtx, {{
            type: 'doughnut',
            data: {{
                labels: {json.dumps(prompt_names)},
                datasets: [{{
                    data: {json.dumps(prompt_percentages)},
                    backgroundColor: [
                        '#3b82f6', '#10b981', '#ef4444', '#f59e0b', '#8b5cf6',
                        '#14b8a6', '#111827', '#ec4899', '#6366f1', '#0ea5e9'
                    ],
                    borderWidth: 0,
                    hoverOffset: 6
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                cutout: '65%',
                plugins: {{
                    legend: {{
                        position: 'right',
                        labels: {{
                            boxWidth: 12,
                            padding: 15,
                            font: {{
                                size: 11
                            }},
                            generateLabels: function(chart) {{
                                const original = Chart.overrides.pie.plugins.legend.labels.generateLabels;
                                const labels = original.call(this, chart);
                                
                                for (let i = 0; i < labels.length; i++) {{
                                    const value = chart.data.datasets[0].data[i];
                                    labels[i].text = chart.data.labels[i] + ': ' + value + '%';
                                }}
                                
                                return labels;
                            }}
                        }}
                    }},
                    tooltip: {{
                        backgroundColor: 'rgba(17, 24, 39, 0.9)',
                        titleColor: '#f3f4f6',
                        bodyColor: '#f3f4f6',
                        padding: 12,
                        cornerRadius: 8,
                        displayColors: false,
                        callbacks: {{
                            label: function(context) {{
                                return context.label + ': ' + context.raw + '%';
                            }}
                        }}
                    }}
                }}
            }}
        }});
        
        // Provider Performance Chart
        const providerCtx = document.getElementById('providerPerformance').getContext('2d');
        new Chart(providerCtx, {{
            type: 'bar',
            data: {{
                labels: {json.dumps(provider_names)},
                datasets: [{{
                    label: 'Interactions',
                    data: {json.dumps(provider_counts)},
                    backgroundColor: [
                        '#3b82f6', '#10b981', '#ef4444', '#f59e0b', '#8b5cf6'
                    ],
                    borderRadius: 8,
                    barThickness: 24,
                    maxBarThickness: 30
                }}]
            }},
            options: {{
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        display: false
                    }},
                    tooltip: {{
                        backgroundColor: 'rgba(17, 24, 39, 0.9)',
                        titleColor: '#f3f4f6',
                        bodyColor: '#f3f4f6',
                        padding: 12,
                        cornerRadius: 8,
                        displayColors: false
                    }}
                }},
                scales: {{
                    x: {{
                        beginAtZero: true,
                        grid: {{
                            display: false
                        }},
                        ticks: {{
                            precision: 0,
                            font: {{
                                size: 11
                            }},
                            color: '#6b7280'
                        }}
                    }},
                    y: {{
                        grid: {{
                            display: false
                        }},
                        ticks: {{
                            font: {{
                                size: 11
                            }},
                            color: '#6b7280'
                        }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""
    
    # Write the HTML content to file
    with open(dashboard_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return dashboard_path