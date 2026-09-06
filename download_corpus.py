"""
Скачивает набор RFC в папку corpus/.
Запуск:  .venv\Scripts\python.exe download_corpus.py
Уже скачанные файлы пропускаются, так что можно запускать повторно.
"""

import os
import requests

# (номер RFC, о чём он) — можно смело удалять строки, если хочешь корпус поменьше
RFCS = [
    (768,  "UDP"),
    (791,  "IPv4"),
    (792,  "ICMP"),
    (826,  "ARP"),
    (1034, "DNS: концепции"),
    (1035, "DNS: реализация"),
    (1122, "Требования к хостам"),
    (1812, "Требования к IPv4-роутерам"),
    (1918, "Приватные адреса"),
    (2131, "DHCP"),
    (2132, "Опции DHCP"),
    (2328, "OSPFv2"),
    (2784, "GRE"),
    (2827, "Ingress filtering (BCP 38)"),
    (4271, "BGP-4"),
    (4291, "Адресация IPv6"),
    (4301, "Архитектура IPsec"),
    (4443, "ICMPv6"),
    (4787, "Поведение NAT для UDP"),
    (4861, "Neighbor Discovery для IPv6"),
    (4862, "SLAAC"),
    (5424, "Syslog"),
    (6598, "Shared Address Space (CGNAT)"),
    (6891, "EDNS(0)"),
    (7296, "IKEv2"),
    (7348, "VXLAN"),
    (8200, "IPv6"),
    (8446, "TLS 1.3"),
    (8484, "DNS over HTTPS"),
    (9000, "QUIC"),
    (9110, "HTTP: семантика"),
    (9112, "HTTP/1.1"),
    (9293, "TCP"),
]

OUT_DIR = "corpus"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    total = 0

    for num, title in RFCS:
        path = os.path.join(OUT_DIR, f"rfc{num}.txt")

        if os.path.exists(path):
            print(f"  уже есть: rfc{num}")
            continue

        url = f"https://www.rfc-editor.org/rfc/rfc{num}.txt"
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
        except Exception as e:
            print(f"  !! rfc{num} не скачался: {e}")
            continue

        resp.encoding = "utf-8"
        text = resp.text

        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

        total += len(text)
        print(f"  rfc{num:<5} {title:<32} {len(text) // 1024} КБ")

    print(f"\nГотово. Скачано за этот запуск: {total // 1024} КБ, папка: {OUT_DIR}/")


if __name__ == "__main__":
    main()
