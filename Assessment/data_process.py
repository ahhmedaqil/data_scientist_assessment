"""JSON to DataFrame"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

with open("conversation_data.json", "r", encoding="utf-8") as file:
    data = json.load(file)

conversations = []
messages = []

for conv_num in data["conversations"]:
  conversations.append({
      "conversation_id": conv_num["conversation_id"],
      "user_id": conv_num["metadata"]["user_id"],
      "timestamp": conv_num["timestamp"],
      "duration": conv_num["duration"],
      "satisfaction_score": conv_num["metadata"]["satisfaction_score"],
      "resolved": conv_num["metadata"]["resolved"],
      "category": conv_num["metadata"]["category"],
      "priority": conv_num["metadata"]["priority"],
      "platform": conv_num["metadata"]["platform"],
      "first_time_user": conv_num["metadata"]["first_time_user"],
      "message_count": len(conv_num["messages"]),
      "tags": ", ".join(conv_num["metadata"].get("tags", [])),
      })

  for msg_num in conv_num["messages"]:
    messages.append({
        "conversation_id": conv_num["conversation_id"],
        "message_id": msg_num["id"],
        "timestamp": msg_num["timestamp"],
        "sender": msg_num["sender"],
        "text": msg_num["text"],
        "response_time": msg_num["response_time"],
        })

df_conversations = pd.DataFrame(conversations)
df_messages = pd.DataFrame(messages)

"""Data Preprocessing and Quality Issues (TimeStamp, Duration and Satisfaction Score)"""

# timestamp to datetime format
df_conversations["timestamp"] = pd.to_datetime(df_conversations["timestamp"], utc=True)
df_messages["timestamp"] = pd.to_datetime(df_messages["timestamp"], utc=True)

# if duration is null, calculate it
for index, row in df_conversations.iterrows():
  if pd.isnull(row["duration"]):
    conv_id = row["conversation_id"]
    conv_messages = df_messages[df_messages["conversation_id"] == conv_id]

    if not conv_messages.empty:
      start_time = conv_messages["timestamp"].min()
      end_time = conv_messages["timestamp"].max()
      duration = (end_time - start_time).total_seconds()
      df_conversations.at[index, "duration"] = duration

# if satisfaction_score is NaN, we will try to infer score based on conversation outcome
mean_resolved = df_conversations[df_conversations["resolved"] == True]["satisfaction_score"].mean()
mean_unresolved = df_conversations[df_conversations["resolved"] == False]["satisfaction_score"].mean()

df_conversations.loc[(df_conversations["satisfaction_score"].isna()) & (df_conversations["resolved"] == True), "satisfaction_score"] = mean_resolved
df_conversations.loc[(df_conversations["satisfaction_score"].isna()) & (df_conversations["resolved"] == False), "satisfaction_score"] = mean_unresolved

# flag if message text is missing
df_messages["is_missing"] = df_messages["text"].isna()

# flag if message text is duplicated
df_messages["is_duplicate"] = df_messages.duplicated(subset=["conversation_id", "text"], keep='first')

"""Export processed data to a queryable format"""

import pyarrow

df_conversations.to_parquet("conversations.parquet", engine="pyarrow", index=False)
df_messages.to_parquet("messages.parquet", engine="pyarrow", index=False)

"""Metrics by time period (Date and Day of Week)"""

df_conversations["date"] = df_conversations["timestamp"].dt.date
df_conversations["day_of_week"] = df_conversations["timestamp"].dt.day_name()

daily_metrics = df_conversations.groupby("date").agg(
    total_conversations=("conversation_id", "count"),
    avg_duration=("duration", "mean"),
    resolved_rate=("resolved", "mean"),
    avg_satisfaction=("satisfaction_score", "mean"),
)

weekly_metrics = df_conversations.groupby("day_of_week").agg(
    total_conversations=("conversation_id", "count"),
    avg_duration=("duration", "mean"),
    resolved_rate=("resolved", "mean"),
    avg_satisfaction=("satisfaction_score", "mean"),
).reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])

print("Daily Metrics:\n", daily_metrics)
print("\nWeekly Metrics:\n", weekly_metrics)

fig, ax1 = plt.subplots(figsize=(12, 6))

ax1.set_title("Daily Metrics Over Time", fontsize=14)
ax1.set_xlabel("Date")

ax1.set_ylabel("Avg Duration (seconds)", color="tab:blue")
ax1.plot(daily_metrics.index, daily_metrics["avg_duration"], label="Avg Duration", color="tab:blue", marker="o")
ax1.tick_params(axis="y", labelcolor="tab:blue")

ax2 = ax1.twinx()
ax2.set_ylabel("Avg Satisfaction Score", color="tab:green")
ax2.plot(daily_metrics.index, daily_metrics["avg_satisfaction"], label="Avg Satisfaction", color="tab:green", linestyle="dashed", marker="s")
ax2.tick_params(axis="y", labelcolor="tab:green")

fig.tight_layout()
plt.savefig("images/daily_metrics_plot.png")
plt.close()

"""Average Response Time / Conversation"""

df_responses_by_agent = df_messages[df_messages["sender"] == "agent"]

response_time_per_conv = df_responses_by_agent.groupby("conversation_id")["response_time"].mean().reset_index()
response_time_per_conv.rename(columns={"response_time": "avg_response_time"}, inplace=True)

print(response_time_per_conv)

"""Message Count Distribution"""

message_count_dist = df_messages.groupby("conversation_id")["message_id"].count().reset_index()
message_count_dist.rename(columns={"message_id": "message_count"}, inplace=True)

print(message_count_dist)

sns.set_style("whitegrid")

plt.figure(figsize=(10, 5))

sns.histplot(message_count_dist["message_count"], bins=20, kde=True, color="royalblue")

plt.xlabel("Message Count per Conversation")
plt.ylabel("Frequency")
plt.title("Distribution of Message Counts per Conversation")

plt.savefig("images/message_count_distribution.png")
plt.close()

"""Conversation Duration Statistics"""

import numpy as np

durations = df_conversations["duration"]

duration_stats = {
    "Mean Duration (seconds)": np.mean(durations),
    "Median Duration (seconds)": np.median(durations),
    "Min Duration (seconds)": np.min(durations),
    "Max Duration (seconds)": np.max(durations),
    "Standard Deviation": np.std(durations),
    "90th Percentile": np.percentile(durations, 90),
    "95th Percentile": np.percentile(durations, 95),
}

print(duration_stats)

"""Resolution Rate / Time"""

res_rate = df_conversations.groupby("date").agg(total_conversations=("conversation_id", "count"), resolved_conversations=("resolved", "sum")).reset_index()
res_rate["resolution_rate"] = (res_rate["resolved_conversations"] / res_rate["total_conversations"]) * 100

print(res_rate)

"""User Satisfaction Trend"""

# user satisfaction by platform
satisfaction_by_platform = df_conversations.groupby(["date", "platform"])["satisfaction_score"].mean().reset_index()

print(satisfaction_by_platform)

# user satisfaction by use
satisfaction_by_use = df_conversations.groupby(["date", "first_time_user"])["satisfaction_score"].mean().reset_index()

print(satisfaction_by_use)

# user satisfaction by category
satisfaction_by_category = df_conversations.groupby(["date", "category"])["satisfaction_score"].mean().reset_index()

print(satisfaction_by_category)

# user satisfaction by number of messages
satisfaction_by_msg = df_conversations.groupby(["message_count"])["satisfaction_score"].mean().reset_index()

print(satisfaction_by_msg)

# user satisfaction by number of messages / day
satisfaction_by_msg_day = df_conversations.groupby(["date", "message_count"])["satisfaction_score"].mean().reset_index()

print(satisfaction_by_msg_day)

"""Peak Conversation Times"""

df_conversations["hour"] = df_conversations["timestamp"].dt.hour

peak_times = (df_conversations.groupby(["date", "hour"]).size().reset_index(name="conversation_count"))

print(peak_times)

"""Common Conversation Patterns"""

category_resolved_summary = df_conversations.groupby("category")["resolved"].value_counts().unstack(fill_value=0)

print(category_resolved_summary)

df_conversation_expanded = df_conversations.assign(tags=df_conversations["tags"].str.split(", ")).explode("tags")
tags_ag_category = df_conversation_expanded.groupby("tags")["category"].value_counts().unstack(fill_value=0)

print(tags_ag_category)

# time series of key metrics
df_responses_by_agent["date"] = df_responses_by_agent["timestamp"].dt.date

response_time_per_conv = df_responses_by_agent.groupby("date")["response_time"].mean().reset_index()
response_time_per_conv.rename(columns={"response_time": "avg_response_time"}, inplace=True)

plt.figure(figsize=(12, 6))
plt.plot(response_time_per_conv["date"], response_time_per_conv["avg_response_time"], marker='o', linestyle='-', color='blue')

plt.xlabel("Date")
plt.ylabel("Average Response Time (seconds)")
plt.title("Average Agent Response Time")
plt.xticks(rotation=45)
plt.grid(True)

plt.savefig("images/avg_res_time.png")
plt.close()

# distribution plot
plt.figure(figsize=(10, 6))
sns.histplot(df_conversations['duration'], bins=30, kde=True, color='blue')

plt.xlabel("Conversation Duration (seconds)")
plt.ylabel("Frequency")
plt.title("Distribution of Conversation Durations")

plt.savefig("images/conv_count_dist.png")
plt.close()

# summary statistic
category_resolved_summary = category_resolved_summary.reset_index()
category_resolved_summary.rename(columns={False: "Unresolved", True: "Resolved"}, inplace=True)
category_resolved_summary.set_index("category")[["Unresolved", "Resolved"]].plot(kind="bar", stacked=True, color=["red", "green"], figsize=(12, 6))

plt.xlabel("Category")
plt.ylabel("Number of Conversations")
plt.title("Resolved vs Unresolved Conversations by Category")
plt.xticks(rotation=45)
plt.legend(title="Resolution Status")

plt.savefig("images/resolve_count.png")
plt.close()

plt.figure(figsize=(12, 6))
sns.heatmap(tags_ag_category, annot=True, cmap="coolwarm", linewidths=0.5, fmt="d")
plt.title("Category vs. Tags Heatmap")
plt.xlabel("Category")
plt.ylabel("Tags")

plt.savefig("images/tag_category.png")
plt.close()

# data quality metric
missing_values_df = pd.DataFrame(conversations)

missing_percent = (missing_values_df.isnull().sum() / len(missing_values_df)) * 100

plt.figure(figsize=(10, 6))
missing_percent.plot(kind='bar', color='red', alpha=0.7)
plt.xlabel("Columns")
plt.ylabel("Percentage Missing")
plt.title("Missing Data Percentage per Column")
plt.xticks(rotation=45)

plt.savefig("images/miss_data_col.png")
plt.close()
