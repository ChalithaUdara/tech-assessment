"""Streamlit Analytics Dashboard for Datacom AI Platform.

This dashboard visualizes:
- Latency and cost metrics over time
- Retrieval accuracy curves
- Agent success/failure breakdown
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from datacom_ai.utils.log_parser import (
    filter_events_by_date_range,
    filter_events_by_type,
    parse_log_file,
)

# Page configuration
st.set_page_config(
    page_title="Datacom AI Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Default log file path
DEFAULT_LOG_PATH = "logs/chat.jsonl"


@st.cache_data
def load_log_data(log_path: str):
    """
    Load and parse log data with caching.
    
    Args:
        log_path: Path to the log file
        
    Returns:
        List of parsed events
    """
    try:
        events = parse_log_file(log_path)
        return events
    except FileNotFoundError:
        st.error(f"Log file not found: {log_path}")
        return []
    except Exception as e:
        st.error(f"Error loading log file: {str(e)}")
        return []


def main():
    """Main dashboard application."""
    st.title("📊 Datacom AI Analytics Dashboard")
    st.markdown("Visualize latency, cost, retrieval accuracy, and agent performance metrics")
    
    # Sidebar for filters
    with st.sidebar:
        st.header("Filters")
        
        # Log file path selector
        log_path = st.text_input(
            "Log File Path",
            value=DEFAULT_LOG_PATH,
            help="Path to the JSONL log file"
        )
        
        # Date range filter
        st.subheader("Date Range")
        use_date_filter = st.checkbox("Filter by date range", value=False)
        
        start_date = None
        end_date = None
        if use_date_filter:
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Start Date",
                    value=datetime.now().date() - timedelta(days=7),
                )
            with col2:
                end_date = st.date_input(
                    "End Date",
                    value=datetime.now().date(),
                )
            
            if start_date > end_date:
                st.error("Start date must be before end date")
                return
        
        # Convert date inputs to datetime
        if start_date:
            start_date = datetime.combine(start_date, datetime.min.time())
        if end_date:
            # Set to end of day to make it inclusive
            from datetime import time as dt_time
            end_date = datetime.combine(end_date, dt_time.max)
        
        st.divider()
        
        # Refresh button
        if st.button("🔄 Refresh Data", width='stretch'):
            st.cache_data.clear()
            st.rerun()
    
    # Load data
    with st.spinner("Loading log data..."):
        events = load_log_data(log_path)
    
    if not events:
        st.warning("No events found in log file. Please check the log file path.")
        return
    
    # Apply date filter if enabled
    if use_date_filter and (start_date or end_date):
        events = filter_events_by_date_range(events, start_date, end_date)
    
    if not events:
        st.warning("No events found in the selected date range.")
        return
    
    st.sidebar.metric("Total Events", len(events))
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs([
        "📈 Latency & Cost",
        "🎯 Retrieval Accuracy",
        "🤖 Agent Performance"
    ])
    
    # Tab 1: Latency and Cost Metrics
    with tab1:
        render_latency_cost_section(events)
    
    # Tab 2: Retrieval Accuracy
    with tab2:
        render_retrieval_accuracy_section(events)
    
    # Tab 3: Agent Performance
    with tab3:
        render_agent_performance_section(events)


def render_latency_cost_section(events: list):
    """Render latency and cost metrics section."""
    st.header("Latency and Cost Metrics Over Time")
    
    # Get relevant events
    chat_response_events = filter_events_by_type(events, "chat_response")
    llm_call_events = filter_events_by_type(events, "llm_call")
    
    if not chat_response_events and not llm_call_events:
        st.info("No latency or cost data available.")
        return
    
    # Time granularity selector
    col1, col2 = st.columns([1, 3])
    with col1:
        granularity = st.selectbox(
            "Time Granularity",
            ["Hourly", "Daily"],
            index=1,
        )
    
    # Combine events for analysis
    all_events = []
    for event in chat_response_events + llm_call_events:
        if event.get("timestamp") and event.get("latency_ms") is not None:
            all_events.append({
                "timestamp": event["timestamp"],
                "latency_ms": event.get("latency_ms", 0),
                "cost_usd": event.get("cost_usd", 0),
                "event_type": event.get("event_type", "unknown"),
            })
    
    if not all_events:
        st.info("No events with latency or cost data available.")
        return
    
    # Create DataFrame
    df = pd.DataFrame(all_events)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Aggregate by time granularity
    if granularity == "Hourly":
        df["time_bucket"] = df["timestamp"].dt.floor("H")
        freq = "H"
    else:  # Daily
        df["time_bucket"] = df["timestamp"].dt.floor("D")
        freq = "D"
    
    # Aggregate metrics
    aggregated = df.groupby("time_bucket").agg({
        "latency_ms": ["mean", "count"],
        "cost_usd": "sum",
    }).reset_index()
    
    aggregated.columns = ["time_bucket", "avg_latency_ms", "request_count", "total_cost_usd"]
    
    # Sort by time bucket for proper display
    aggregated = aggregated.sort_values("time_bucket").reset_index(drop=True)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Requests", len(all_events))
    with col2:
        avg_latency = df["latency_ms"].mean()
        st.metric("Avg Latency", f"{avg_latency:.0f} ms")
    with col3:
        total_cost = df["cost_usd"].sum()
        st.metric("Total Cost", f"${total_cost:.6f}")
    with col4:
        avg_cost_per_request = total_cost / len(all_events) if all_events else 0
        st.metric("Avg Cost/Request", f"${avg_cost_per_request:.6f}")
    
    st.divider()
    
    # Create charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Latency chart
        if len(aggregated) > 0:
            fig_latency = px.line(
                aggregated,
                x="time_bucket",
                y="avg_latency_ms",
                title="Average Latency Over Time",
                labels={
                    "time_bucket": "Time",
                    "avg_latency_ms": "Latency (ms)",
                },
                markers=True,
            )
            fig_latency.update_traces(
                line=dict(width=2),
                marker=dict(size=6),
            )
            fig_latency.update_layout(
                xaxis_title="Time",
                yaxis_title="Latency (ms)",
                hovermode="x unified",
                showlegend=False,
            )
            st.plotly_chart(fig_latency, width='stretch')
        else:
            st.info("No latency data available for the selected time range.")
    
    with col2:
        # Cost chart
        if len(aggregated) > 0:
            fig_cost = px.line(
                aggregated,
                x="time_bucket",
                y="total_cost_usd",
                title="Total Cost Over Time",
                labels={
                    "time_bucket": "Time",
                    "total_cost_usd": "Cost (USD)",
                },
                markers=True,
            )
            fig_cost.update_traces(
                line=dict(width=2, color="red"),
                marker=dict(size=6, color="red"),
            )
            fig_cost.update_layout(
                xaxis_title="Time",
                yaxis_title="Cost (USD)",
                hovermode="x unified",
                showlegend=False,
            )
            st.plotly_chart(fig_cost, width='stretch')
        else:
            st.info("No cost data available for the selected time range.")


def render_retrieval_accuracy_section(events: list):
    """Render retrieval accuracy curves section."""
    st.header("Retrieval Accuracy Curves")
    
    # Get RAG retrieval events with accuracy metrics
    rag_events = filter_events_by_type(events, "rag_retrieval")
    
    # Filter events that have accuracy metrics
    accuracy_events = []
    available_metrics = set()
    
    for event in rag_events:
        if not event.get("timestamp"):
            continue
        
        # Check which metrics are available
        event_metrics = {}
        metric_fields = [
            "answer_relevancy",
            "faithfulness",
            "contextual_precision",
            "contextual_recall",
            "contextual_relevancy",
        ]
        
        for metric in metric_fields:
            if metric in event and event[metric] is not None:
                event_metrics[metric] = event[metric]
                available_metrics.add(metric)
        
        if event_metrics:
            accuracy_events.append({
                "timestamp": event["timestamp"],
                **event_metrics,
            })
    
    if not accuracy_events:
        st.info("No retrieval accuracy metrics available in the logs.")
        return
    
    # Metric selector
    selected_metrics = st.multiselect(
        "Select Metrics to Display",
        options=sorted(available_metrics),
        default=sorted(available_metrics)[:2] if len(available_metrics) >= 2 else sorted(available_metrics),
    )
    
    if not selected_metrics:
        st.warning("Please select at least one metric to display.")
        return
    
    # Create DataFrame
    df = pd.DataFrame(accuracy_events)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    
    # Summary statistics
    st.subheader("Summary Statistics")
    stats_cols = st.columns(len(selected_metrics))
    
    for idx, metric in enumerate(selected_metrics):
        with stats_cols[idx]:
            metric_values = df[metric].dropna()
            if len(metric_values) > 0:
                st.metric(
                    metric.replace("_", " ").title(),
                    f"{metric_values.mean():.3f}",
                    delta=f"Range: {metric_values.min():.3f} - {metric_values.max():.3f}",
                )
    
    st.divider()
    
    # Time series chart
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set1
    
    for idx, metric in enumerate(selected_metrics):
        metric_df = df[["timestamp", metric]].dropna()
        if len(metric_df) > 0:
            fig.add_trace(
                go.Scatter(
                    x=metric_df["timestamp"],
                    y=metric_df[metric],
                    name=metric.replace("_", " ").title(),
                    mode="lines+markers",
                    line=dict(color=colors[idx % len(colors)]),
                )
            )
    
    fig.update_layout(
        title="Retrieval Accuracy Metrics Over Time",
        xaxis_title="Time",
        yaxis_title="Score",
        hovermode="x unified",
        yaxis=dict(range=[0, 1]),  # Accuracy scores are typically 0-1
    )
    
    st.plotly_chart(fig, width='stretch')
    
    # Data table
    with st.expander("View Raw Data"):
        display_df = df[["timestamp"] + selected_metrics].copy()
        display_df["timestamp"] = display_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
        st.dataframe(display_df, width='stretch')


def render_agent_performance_section(events: list):
    """Render agent success/failure breakdown section."""
    st.header("Agent Success/Failure Breakdown")
    
    # Get agent execution events
    agent_events = filter_events_by_type(events, "agent_execution")
    
    if not agent_events:
        st.info("No agent execution events found in the logs.")
        return
    
    # Create DataFrame
    agent_data = []
    for event in agent_events:
        if event.get("timestamp"):
            agent_data.append({
                "timestamp": event["timestamp"],
                "agent_type": event.get("agent_type", "Unknown"),
                "overall_status": event.get("overall_status", "unknown"),
                "total_execution_time_ms": event.get("total_execution_time_ms", 0),
                "steps_succeeded": event.get("steps_succeeded", 0),
                "steps_failed": event.get("steps_failed", 0),
                "failure_step": event.get("failure_step"),
                "failure_type": event.get("failure_type"),
            })
    
    if not agent_data:
        st.info("No agent execution data with timestamps available.")
        return
    
    df = pd.DataFrame(agent_data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Executions", len(df))
    with col2:
        success_count = len(df[df["overall_status"] == "success"])
        success_rate = (success_count / len(df)) * 100 if len(df) > 0 else 0
        st.metric("Success Rate", f"{success_rate:.1f}%")
    with col3:
        failure_count = len(df[df["overall_status"] == "failure"])
        failure_rate = (failure_count / len(df)) * 100 if len(df) > 0 else 0
        st.metric("Failure Rate", f"{failure_rate:.1f}%")
    with col4:
        avg_execution_time = df["total_execution_time_ms"].mean()
        st.metric("Avg Execution Time", f"{avg_execution_time:.0f} ms")
    
    st.divider()
    
    # Agent type filter
    agent_types = sorted(df["agent_type"].unique())
    selected_agent_types = st.multiselect(
        "Filter by Agent Type",
        options=agent_types,
        default=agent_types,
    )
    
    if selected_agent_types:
        df_filtered = df[df["agent_type"].isin(selected_agent_types)]
    else:
        df_filtered = df
    
    if len(df_filtered) == 0:
        st.warning("No data available for selected agent types.")
        return
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Overall status pie chart
        status_counts = df_filtered["overall_status"].value_counts()
        fig_pie = px.pie(
            values=status_counts.values,
            names=status_counts.index,
            title="Overall Status Distribution",
        )
        st.plotly_chart(fig_pie, width='stretch')
    
    with col2:
        # Success/failure by agent type
        agent_status = (
            df_filtered.groupby(["agent_type", "overall_status"])
            .size()
            .reset_index(name="count")
        )
        
        fig_bar = px.bar(
            agent_status,
            x="agent_type",
            y="count",
            color="overall_status",
            title="Success/Failure by Agent Type",
            labels={"agent_type": "Agent Type", "count": "Count"},
        )
        st.plotly_chart(fig_bar, width='stretch')
    
    # Success rate over time
    # Use datetime floor to group by day for better compatibility
    df_filtered["date"] = df_filtered["timestamp"].dt.floor("D")
    daily_stats = df_filtered.groupby("date").agg({
        "overall_status": lambda x: (x == "success").sum() / len(x) * 100 if len(x) > 0 else 0,
    }).reset_index()
    daily_stats.columns = ["date", "success_rate"]
    daily_stats = daily_stats.sort_values("date").reset_index(drop=True)
    
    if len(daily_stats) > 0:
        fig_timeseries = px.line(
            daily_stats,
            x="date",
            y="success_rate",
            title="Success Rate Over Time",
            labels={"date": "Date", "success_rate": "Success Rate (%)"},
            markers=True,
        )
        fig_timeseries.update_traces(
            line=dict(width=2),
            marker=dict(size=6),
        )
        fig_timeseries.update_layout(
            yaxis=dict(range=[0, 100]),
            hovermode="x unified",
            xaxis_title="Date",
            yaxis_title="Success Rate (%)",
        )
        st.plotly_chart(fig_timeseries, width='stretch')
    else:
        st.info("No data available for success rate over time.")
    
    # Failure breakdown
    if "failure_type" in df_filtered.columns:
        failure_df = df_filtered[df_filtered["overall_status"] == "failure"]
        if len(failure_df) > 0:
            failure_types = failure_df["failure_type"].dropna()
            if len(failure_types) > 0:
                st.subheader("Failure Type Breakdown")
                failure_counts = failure_types.value_counts()
                fig_failure = px.bar(
                    x=failure_counts.index,
                    y=failure_counts.values,
                    title="Failure Types",
                    labels={"x": "Failure Type", "y": "Count"},
                )
                st.plotly_chart(fig_failure, width='stretch')
    
    # Data table
    with st.expander("View Raw Data"):
        display_df = df_filtered.copy()
        display_df["timestamp"] = display_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
        st.dataframe(display_df, width='stretch')


if __name__ == "__main__":
    main()

