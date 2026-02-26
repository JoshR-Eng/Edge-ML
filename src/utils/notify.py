"""
DESCRIPTION:
    This module allows a notification to be sent to a 
    discord channel using a webhook URL.

USAGE:
    - Store the webhook URL in an environment variable named 'DISCORD_WEBHOOK_URL'.
    - Call the 'send_notification' function with the desired message.
"""


#=============================================================  
#                           IMPORTS
#=============================================================  

import os
import urllib.request
import json


#=============================================================
#                          FUNCTIONS
#=============================================================

def send_notification(message):

    # Fetch the secret webhook URL from environment variables
    try:
        webhook_url = os.getenv('DISCORD_WEBHOOK_URL')
    except Exception as e:
        print(f"Error fetching webhook URL: {e}")
        return

    if not webhook_url:
        print("No Discord webhook URL found in environment.")
        return

    payload = {
        "content": message,
        "username": "Jetson Orin Nano"
    }

    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(webhook_url, data=data, method='POST')
    req.add_header('Content-Type', 'application/json')
    req.add_header('User-Agent', 'JetsonBenchmark/1.0')

    try:
        urllib.request.urlopen(req)
        print("Notification sent successfully.")
    except Exception as e:
        print(f"Error sending notification: {e}")
