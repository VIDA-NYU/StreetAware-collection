# syntax=docker/dockerfile:1
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        openssh-server \
        python3 \
        python3-minimal && \
    rm -rf /var/lib/apt/lists/*

RUN useradd -m sensor && echo 'sensor:sensor' | chpasswd && \
    mkdir -p /var/run/sshd

COPY log_writer.py /opt/mock-sensor/log_writer.py
RUN chmod +x /opt/mock-sensor/log_writer.py

EXPOSE 22

CMD ["/usr/sbin/sshd", "-D"]
