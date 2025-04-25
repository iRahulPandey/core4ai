"""
Core4AI Analytics Module

This module provides analytics tracking and reporting capabilities for Core4AI.
"""

import sqlite3
import json
import time
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

# Create logger
logger = logging.getLogger("core4ai.analytics")

# Import config directory path
from ..config.config import CONFIG_DIR, get_analytics_config

# Analytics database file
ANALYTICS_DB = None  # Will be set dynamically based on config

def get_analytics_db_path():
    """Get the analytics database path from config or use default."""
    global ANALYTICS_DB
    
    # Only compute once
    if ANALYTICS_DB is not None:
        return ANALYTICS_DB
        
    # Get analytics config
    analytics_config = get_analytics_config()
    
    # Use configured path if analytics is enabled
    if analytics_config.get('enabled', False) and analytics_config.get('db_path'):
        ANALYTICS_DB = Path(analytics_config['db_path'])
    else:
        # Default location
        ANALYTICS_DB = CONFIG_DIR / "analytics.db"
        
    return ANALYTICS_DB

def is_analytics_enabled():
    """Check if analytics is enabled in the configuration."""
    analytics_config = get_analytics_config()
    return analytics_config.get('enabled', False)

def ensure_analytics_db():
    """Ensure analytics database exists and has correct schema."""
    # Skip if analytics is disabled
    if not is_analytics_enabled():
        logger.info("Analytics is disabled. Skipping database initialization.")
        return False
        
    try:
        # Make sure the config directory exists
        db_path = get_analytics_db_path()
        os.makedirs(db_path.parent, exist_ok=True)
            
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create prompt_usage table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS prompt_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp INTEGER NOT NULL,
            prompt_name TEXT NOT NULL,
            prompt_version INTEGER,
            confidence REAL,
            duration REAL,
            successful INTEGER,
            parameters TEXT,
            metadata TEXT
        )
        ''')
        
        # Create summary metrics table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS prompt_metrics (
            prompt_name TEXT NOT NULL,
            prompt_version INTEGER NOT NULL,
            total_uses INTEGER DEFAULT 0,
            avg_confidence REAL DEFAULT 0,
            avg_duration REAL DEFAULT 0,
            success_rate REAL DEFAULT 0,
            last_used INTEGER,
            PRIMARY KEY (prompt_name, prompt_version)
        )
        ''')
        
        conn.commit()
        conn.close()
        
        logger.info(f"Analytics database initialized at {db_path}")
        return True
    except Exception as e:
        logger.error(f"Error initializing analytics database: {e}")
        return False

def record_prompt_usage(
    prompt_name: str, 
    prompt_version: Optional[int] = None,
    confidence: Optional[float] = None,
    duration: Optional[float] = None,
    successful: bool = True,
    parameters: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Record usage of a prompt in the analytics database.
    
    Args:
        prompt_name: Name of the prompt that was used
        prompt_version: Version of the prompt that was used
        confidence: Confidence score of the match (0-100)
        duration: Duration of processing in seconds
        successful: Whether the prompt was used successfully
        parameters: Parameters used with the prompt
        metadata: Additional metadata about usage (provider, model, etc.)
        
    Returns:
        True if recording was successful, False otherwise
    """
    # Skip if analytics is disabled
    if not is_analytics_enabled():
        return False
    
    # Ensure database exists
    ensure_analytics_db()
    
    try:
        conn = sqlite3.connect(get_analytics_db_path())
        cursor = conn.cursor()
        
        # Current timestamp
        current_time = int(time.time())
        
        # Convert dictionaries to JSON strings
        params_json = json.dumps(parameters) if parameters else None
        metadata_json = json.dumps(metadata) if metadata else None
        
        # Default version to 1 if not specified
        if prompt_version is None:
            prompt_version = 1
        
        # Convert bool to int for SQLite
        successful_int = 1 if successful else 0
        
        # Insert usage record
        cursor.execute(
            '''
            INSERT INTO prompt_usage 
            (timestamp, prompt_name, prompt_version, confidence, duration, 
            successful, parameters, metadata) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            (current_time, prompt_name, prompt_version, confidence, duration, 
            successful_int, params_json, metadata_json)
        )
        
        # Update metrics table
        cursor.execute(
            '''
            INSERT INTO prompt_metrics 
            (prompt_name, prompt_version, total_uses, avg_confidence, 
            avg_duration, success_rate, last_used)
            VALUES (?, ?, 1, ?, ?, ?, ?)
            ON CONFLICT(prompt_name, prompt_version) DO UPDATE SET
            total_uses = total_uses + 1,
            avg_confidence = ((avg_confidence * (total_uses - 1)) + IFNULL(?, 0)) / total_uses,
            avg_duration = ((avg_duration * (total_uses - 1)) + IFNULL(?, 0)) / total_uses,
            success_rate = ((success_rate * (total_uses - 1)) + ?) / total_uses,
            last_used = ?
            ''',
            (prompt_name, prompt_version, confidence, duration, successful_int, current_time,
             confidence, duration, successful_int, current_time)
        )
        
        conn.commit()
        conn.close()
        
        logger.debug(f"Recorded usage of prompt '{prompt_name}' v{prompt_version}")
        return True
    except Exception as e:
        logger.error(f"Error recording prompt usage: {e}")
        return False

def get_prompt_analytics(
    prompt_name: Optional[str] = None,
    time_range: Optional[int] = None,
    version: Optional[int] = None,
    limit: int = 100
) -> Dict[str, Any]:
    """
    Get analytics data for prompts.
    
    Args:
        prompt_name: Name of the specific prompt to analyze (None for all)
        time_range: Time range in days (None for all time)
        version: Specific version to filter by (None for all versions)
        limit: Maximum number of records to return
        
    Returns:
        Dictionary with analytics data
    """
    # Check if analytics is enabled
    if not is_analytics_enabled():
        return {
            "status": "error",
            "error": "Analytics is disabled. Enable it in the configuration."
        }
    
    # Ensure database exists
    ensure_analytics_db()
    
    try:
        conn = sqlite3.connect(get_analytics_db_path())
        conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        cursor = conn.cursor()
        
        # Calculate timestamp for time range
        time_filter = None
        if time_range:
            time_filter = int(time.time()) - (time_range * 24 * 60 * 60)
            
        # Build query
        query = "SELECT * FROM prompt_usage WHERE 1=1"
        params = []
        
        # Add filters
        if prompt_name:
            query += " AND prompt_name = ?"
            params.append(prompt_name)
            
        if version:
            query += " AND prompt_version = ?"
            params.append(version)
            
        if time_filter:
            query += " AND timestamp >= ?"
            params.append(time_filter)
            
        # Add ordering and limit
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        # Execute query
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        # Process results
        usage_data = []
        for row in rows:
            record = dict(row)
            
            # Parse JSON fields
            if record.get('parameters'):
                try:
                    record['parameters'] = json.loads(record['parameters'])
                except json.JSONDecodeError:
                    record['parameters'] = {}
                    
            if record.get('metadata'):
                try:
                    record['metadata'] = json.loads(record['metadata'])
                except json.JSONDecodeError:
                    record['metadata'] = {}
            
            # Convert timestamp to readable format
            record['datetime'] = datetime.fromtimestamp(
                record['timestamp']
            ).strftime('%Y-%m-%d %H:%M:%S')
            
            # Convert successful back to boolean
            record['successful'] = bool(record['successful'])
            
            usage_data.append(record)
            
        # Get summary metrics
        metrics_query = """
        SELECT * FROM prompt_metrics WHERE 1=1
        """
        metrics_params = []
        
        if prompt_name:
            metrics_query += " AND prompt_name = ?"
            metrics_params.append(prompt_name)
            
        if version:
            metrics_query += " AND prompt_version = ?"
            metrics_params.append(version)
            
        metrics_query += " ORDER BY prompt_name, prompt_version"
        
        cursor.execute(metrics_query, metrics_params)
        metrics = []
        
        for row in cursor.fetchall():
            metric = dict(row)
            
            # Convert last_used to readable datetime
            if metric.get('last_used'):
                metric['last_used_datetime'] = datetime.fromtimestamp(
                    metric['last_used']
                ).strftime('%Y-%m-%d %H:%M:%S')
                
            metrics.append(metric)
            
        # Get version comparison for specific prompt
        version_comparison = []
        if prompt_name and not version:
            cursor.execute(
                """
                SELECT prompt_version, total_uses, avg_confidence, avg_duration, success_rate
                FROM prompt_metrics
                WHERE prompt_name = ?
                ORDER BY prompt_version DESC
                """,
                (prompt_name,)
            )
            version_comparison = [dict(row) for row in cursor.fetchall()]
            
        # Get most used prompts
        most_used = []
        if not prompt_name:
            cursor.execute(
                """
                SELECT prompt_name, SUM(total_uses) as total_uses, 
                       AVG(avg_confidence) as avg_confidence,
                       AVG(success_rate) as success_rate
                FROM prompt_metrics
                GROUP BY prompt_name
                ORDER BY total_uses DESC
                LIMIT 10
                """
            )
            most_used = [dict(row) for row in cursor.fetchall()]
            
        # Get usage over time
        usage_by_date = []
        if time_filter:
            cursor.execute(
                """
                SELECT strftime('%Y-%m-%d', datetime(timestamp, 'unixepoch')) as date,
                       COUNT(*) as count
                FROM prompt_usage
                WHERE timestamp >= ?
                GROUP BY date
                ORDER BY date
                """,
                (time_filter,)
            )
            usage_by_date = [dict(row) for row in cursor.fetchall()]
            
        # Get usage by provider
        provider_usage = []
        provider_query = """
        SELECT json_extract(metadata, '$.provider') as provider,
               json_extract(metadata, '$.model') as model,
               COUNT(*) as count,
               AVG(confidence) as avg_confidence,
               AVG(duration) as avg_duration,
               SUM(CASE WHEN successful = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as success_rate
        FROM prompt_usage
        WHERE metadata IS NOT NULL
        """
        
        provider_params = []
        if time_filter:
            provider_query += " AND timestamp >= ?"
            provider_params.append(time_filter)
            
        if prompt_name:
            provider_query += " AND prompt_name = ?"
            provider_params.append(prompt_name)
            
        provider_query += " GROUP BY provider, model ORDER BY count DESC"
        
        cursor.execute(provider_query, provider_params)
        provider_usage = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        
        return {
            "status": "success",
            "usage_data": usage_data,
            "metrics": metrics,
            "version_comparison": version_comparison,
            "most_used_prompts": most_used,
            "usage_by_date": usage_by_date,
            "provider_usage": provider_usage,
            "count": len(usage_data),
            "prompt_name": prompt_name,
            "time_range": time_range,
            "version": version
        }
    except Exception as e:
        logger.error(f"Error getting prompt analytics: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

def get_usage_stats(time_range: Optional[int] = None) -> Dict[str, Any]:
    """
    Get overall usage statistics.
    
    Args:
        time_range: Time range in days (None for all time)
        
    Returns:
        Dictionary with usage statistics
    """
    # Check if analytics is enabled
    if not is_analytics_enabled():
        return {
            "status": "error",
            "error": "Analytics is disabled. Enable it in the configuration."
        }
    
    # Ensure database exists
    ensure_analytics_db()
    
    try:
        conn = sqlite3.connect(get_analytics_db_path())
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Calculate timestamp for time range
        time_filter = None
        if time_range:
            time_filter = int(time.time()) - (time_range * 24 * 60 * 60)
            
        # Base query parts
        base_query = "FROM prompt_usage WHERE 1=1"
        params = []
        
        if time_filter:
            base_query += " AND timestamp >= ?"
            params.append(time_filter)
            
        # Get total usage count
        cursor.execute(f"SELECT COUNT(*) as count {base_query}", params)
        total_count = cursor.fetchone()['count']
        
        # Get usage by prompt
        cursor.execute(
            f"""
            SELECT prompt_name, COUNT(*) as count,
                   AVG(confidence) as avg_confidence,
                   AVG(duration) as avg_duration,
                   SUM(CASE WHEN successful = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as success_rate
            {base_query}
            GROUP BY prompt_name
            ORDER BY count DESC
            LIMIT 20
            """,
            params
        )
        usage_by_prompt = [dict(row) for row in cursor.fetchall()]
        
        # Get usage by day
        day_query = f"""
        SELECT strftime('%Y-%m-%d', datetime(timestamp, 'unixepoch')) as date,
               COUNT(*) as count
        {base_query}
        GROUP BY date
        ORDER BY date
        """
        
        cursor.execute(day_query, params)
        usage_by_day = [dict(row) for row in cursor.fetchall()]
        
        # Get usage by provider
        provider_query = f"""
        SELECT json_extract(metadata, '$.provider') as provider,
               json_extract(metadata, '$.model') as model,
               COUNT(*) as count,
               AVG(confidence) as avg_confidence,
               AVG(duration) as avg_duration,
               SUM(CASE WHEN successful = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as success_rate,
               SUM(CASE WHEN json_extract(metadata, '$.fallback_used') = 1 THEN 1 ELSE 0 END) as fallback_count,
               SUM(CASE WHEN json_extract(metadata, '$.fallback_used') = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as fallback_rate
        FROM prompt_usage
        WHERE metadata IS NOT NULL
        """
        
        provider_params = []
        if time_filter:
            provider_query += " AND timestamp >= ?"
            provider_params.append(time_filter)
            
        provider_query += " GROUP BY provider, model ORDER BY count DESC"
        
        cursor.execute(provider_query, provider_params)
        provider_stats = [dict(row) for row in cursor.fetchall()]
        
        # Get content type stats
        content_query = f"""
        SELECT json_extract(metadata, '$.content_type') as content_type,
               COUNT(*) as count,
               AVG(confidence) as avg_confidence,
               AVG(duration) as avg_duration,
               SUM(CASE WHEN successful = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as success_rate
        FROM prompt_usage
        WHERE metadata IS NOT NULL
        """
        
        content_params = []
        if time_filter:
            content_query += " AND timestamp >= ?"
            content_params.append(time_filter)
            
        content_query += " GROUP BY content_type ORDER BY count DESC"
        
        cursor.execute(content_query, content_params)
        content_stats = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        
        return {
            "status": "success",
            "total_count": total_count,
            "usage_by_prompt": usage_by_prompt,
            "usage_by_day": usage_by_day,
            "provider_stats": provider_stats,
            "content_stats": content_stats,
            "time_range": time_range
        }
    except Exception as e:
        logger.error(f"Error getting usage stats: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

def clear_analytics(prompt_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Clear analytics data.
    
    Args:
        prompt_name: Name of prompt to clear data for (None for all prompts)
        
    Returns:
        Dictionary with operation results
    """
    # Check if analytics is enabled
    if not is_analytics_enabled():
        return {
            "status": "error",
            "error": "Analytics is disabled. Enable it in the configuration."
        }
    
    # Ensure database exists
    ensure_analytics_db()
    
    try:
        conn = sqlite3.connect(get_analytics_db_path())
        cursor = conn.cursor()
        
        if prompt_name:
            # Delete specific prompt data
            cursor.execute("DELETE FROM prompt_usage WHERE prompt_name = ?", (prompt_name,))
            cursor.execute("DELETE FROM prompt_metrics WHERE prompt_name = ?", (prompt_name,))
            message = f"Analytics data cleared for prompt: {prompt_name}"
        else:
            # Delete all data
            cursor.execute("DELETE FROM prompt_usage")
            cursor.execute("DELETE FROM prompt_metrics")
            message = "All analytics data cleared"
            
        rows_affected = cursor.rowcount
        conn.commit()
        conn.close()
        
        logger.info(message)
        
        return {
            "status": "success",
            "message": message,
            "rows_affected": rows_affected
        }
    except Exception as e:
        logger.error(f"Error clearing analytics data: {e}")
        return {
            "status": "error",
            "error": str(e)
        }