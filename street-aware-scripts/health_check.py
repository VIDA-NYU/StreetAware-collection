#!/usr/bin/env python3
import json
import paramiko
import socket
import time

# List of nodes to health‐check (match whatever you used in your SSH scripts)
NODES = [
    {"host": "192.168.0.184", "username": "reip", "password": "reip"},
    {"host": "192.168.0.122", "username": "reip", "password": "reip"},
    {"host": "192.168.0.108", "username": "reip", "password": "reip"},
    {"host": "192.168.0.227", "username": "reip", "password": "reip"},
]

# Timeout for each SSH attempt (seconds)
SSH_TIMEOUT = 5

def check_ssh(node):
    """
    Return True if SSH on node['host'] is accepting connections and credentials work,
    False otherwise. We use a very quick connect attempt.
    """
    host = node["host"]
    username = node["username"]
    password = node["password"]

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        # Attempt to open TCP socket first (faster failure than waiting full Paramiko timeout)
        sock = socket.create_connection((host, 22), timeout=SSH_TIMEOUT)
        sock.close()

        # Now attempt SSH authentication
        client.connect(
            hostname=host,
            username=username,
            password=password,
            timeout=SSH_TIMEOUT,
            banner_timeout=SSH_TIMEOUT,
            auth_timeout=SSH_TIMEOUT,
        )
        return True
    except Exception:
        return False
    finally:
        try:
            client.close()
        except Exception:
            pass

def main():
    statuses = {}
    for node in NODES:
        up = check_ssh(node)
        statuses[node["host"]] = "up" if up else "down"

    # Print as JSON to stdout
    print(json.dumps(statuses))

if __name__ == "__main__":
    main()
