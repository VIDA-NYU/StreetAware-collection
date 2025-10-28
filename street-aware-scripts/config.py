#!/usr/bin/env python3

"""Sensor configuration shared between backend and collection scripts."""

from __future__ import annotations

import copy
import os


_REAL_SENSORS = [
    {
        "name": "sensor-184",
        "host": "192.168.0.184",
        "port": 22,
        "username": "reip",
        "password": "reip",
        "display_name": "Sensor 184",
        "process_match": "filter.py",
        "command": "cd software/reip-pipelines/smart-filter && stdbuf -oL -eL python3 -u filter.py",
    },
    {
        "name": "sensor-122",
        "host": "192.168.0.122",
        "port": 22,
        "username": "reip",
        "password": "reip",
        "display_name": "Sensor 122",
        "process_match": "filter.py",
        "command": "cd software/reip-pipelines/smart-filter && stdbuf -oL -eL python3 -u filter.py",
    },
    {
        "name": "sensor-108",
        "host": "192.168.0.108",
        "port": 22,
        "username": "reip",
        "password": "reip",
        "display_name": "Sensor 108",
        "process_match": "filter.py",
        "command": "cd software/reip-pipelines/smart-filter && stdbuf -oL -eL python3 -u filter.py",
    },
    {
        "name": "sensor-227",
        "host": "192.168.0.227",
        "port": 22,
        "username": "reip",
        "password": "reip",
        "display_name": "Sensor 227",
        "process_match": "filter.py",
        "command": "cd software/reip-pipelines/smart-filter && stdbuf -oL -eL python3 -u filter.py",
    },
]

_MOCK_SENSORS = [
    {
        "name": "mock-2221",
        "host": "localhost",
        "port": 2221,
        "username": "sensor",
        "password": "sensor",
        "display_name": "Mock Sensor 1",
        "process_match": "log_writer.py",
        "command": "stdbuf -oL -eL python3 -u /opt/mock-sensor/log_writer.py --name {display_name}",
    },
    {
        "name": "mock-2222",
        "host": "localhost",
        "port": 2222,
        "username": "sensor",
        "password": "sensor",
        "display_name": "Mock Sensor 2",
        "process_match": "log_writer.py",
        "command": "stdbuf -oL -eL python3 -u /opt/mock-sensor/log_writer.py --name {display_name}",
    },
    {
        "name": "mock-2223",
        "host": "localhost",
        "port": 2223,
        "username": "sensor",
        "password": "sensor",
        "display_name": "Mock Sensor 3",
        "process_match": "log_writer.py",
        "command": "stdbuf -oL -eL python3 -u /opt/mock-sensor/log_writer.py --name {display_name}",
    },
    {
        "name": "mock-2224",
        "host": "localhost",
        "port": 2224,
        "username": "sensor",
        "password": "sensor",
        "display_name": "Mock Sensor 4",
        "process_match": "log_writer.py",
        "command": "stdbuf -oL -eL python3 -u /opt/mock-sensor/log_writer.py --name {display_name}",
    },
]

MODES = {
    "real": {
        "sensors": _REAL_SENSORS,
        "default_command": "cd software/reip-pipelines/smart-filter && stdbuf -oL -eL python3 -u filter.py",
    },
    "mock": {
        "sensors": _MOCK_SENSORS,
        "default_command": "stdbuf -oL -eL python3 -u /opt/mock-sensor/log_writer.py --name {display_name}",
    },
}

def _initial_mode() -> str:
    value = os.environ.get("SENSOR_MODE", "real").strip().lower()
    return value if value in MODES else "real"


CURRENT_MODE = _initial_mode()


def set_sensor_mode(mode: str) -> None:
    if mode not in MODES:
        raise ValueError("Mode must be 'mock' or 'real'")
    global CURRENT_MODE
    CURRENT_MODE = mode


def get_current_mode() -> str:
    return CURRENT_MODE


def _clone_sensors(mode: str) -> list[dict]:
    template = MODES[mode]["sensors"]
    return copy.deepcopy(template)


def get_sensor_config() -> list[dict]:
    sensors = _clone_sensors(CURRENT_MODE)
    for sensor in sensors:
        sensor.setdefault("display_name", _sensor_label(sensor))
        sensor.setdefault("name", sensor["display_name"])
        sensor.setdefault("process_match", "filter.py" if CURRENT_MODE == "real" else "log_writer.py")
    return sensors


def get_public_sensor_config() -> list[dict]:
    public = []
    for sensor in get_sensor_config():
        public.append(
            {
                "name": sensor["name"],
                "display_name": sensor["display_name"],
                "host": sensor["host"],
                "port": sensor.get("port", 22),
                "mode": CURRENT_MODE,
            }
        )
    return public


def get_sensor_command() -> str:
    return MODES[CURRENT_MODE]["default_command"]


def _sensor_label(sensor: dict) -> str:
    return sensor.get("display_name") or sensor.get("name") or f"{sensor['host']}:{sensor.get('port', 22)}"
