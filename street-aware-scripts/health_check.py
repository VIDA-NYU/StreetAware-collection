#!/usr/bin/env python3
import json
import socket

import paramiko

from config import get_current_mode, get_sensor_config

SSH_TIMEOUT = 5


def check_sensor(sensor: dict) -> bool:
    host = sensor["host"] if get_current_mode() == "real" else "localhost"
    port = sensor.get("port", 22)
    username = sensor["username"]
    password = sensor["password"]

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        sock = socket.create_connection((host, port), timeout=SSH_TIMEOUT)
        sock.close()

        client.connect(
            hostname=host,
            port=port,
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


def main() -> None:
    statuses = {}
    for sensor in get_sensor_config():
        display = sensor.get("display_name") or sensor.get("name")
        statuses[display] = {
            "status": "up" if check_sensor(sensor) else "down",
            "host": sensor["host"],
            "port": sensor.get("port", 22),
            "mode": get_current_mode(),
        }

    statuses["__mode"] = get_current_mode()

    print(json.dumps(statuses))

if __name__ == "__main__":
    main()
