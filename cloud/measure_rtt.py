"""
cloud/measure_rtt.py  —  DB role 4, the measurement step

Calls the deployed smart endpoint N times and reports the NETWORK
component of latency, i.e. round-trip minus the server-reported compute
time. Paste the result into config.SMART_NETWORK_MS.

    SMART_ENDPOINT_URL=https://... python -m cloud.measure_rtt
"""

import statistics
import time

import requests

from config import SMART_ENDPOINT_URL

SAMPLE_LAP = {
    "tyre_life": 15, "compound": "MEDIUM", "gap_ahead": 1.2, "gap_behind": 3.0,
    "gap_ahead_delta": -0.2, "air_temp": 30.0, "track_temp": 42.0,
    "rainfall": False, "is_caution": 0.0, "field_pits_last_5": 1,
}


def main(n: int = 50):
    if not SMART_ENDPOINT_URL:
        raise SystemExit("set SMART_ENDPOINT_URL first")

    payload = {"laps": [SAMPLE_LAP] * 8}
    rtts, servers = [], []

    for i in range(n):
        t0 = time.perf_counter()
        r = requests.post(SMART_ENDPOINT_URL, json=payload, timeout=30)
        rtt = (time.perf_counter() - t0) * 1000.0
        r.raise_for_status()
        server_ms = r.json().get("server_ms", 0.0)
        rtts.append(rtt)
        servers.append(server_ms)
        if i == 0:
            print(f"cold start: rtt={rtt:.1f} ms  server={server_ms:.1f} ms")

    warm_rtt = rtts[1:]
    warm_srv = servers[1:]
    net = [a - b for a, b in zip(warm_rtt, warm_srv)]

    print(f"\nwarm samples: {len(warm_rtt)}")
    print(f"  rtt      avg={statistics.mean(warm_rtt):8.2f} ms  "
          f"p95={sorted(warm_rtt)[int(0.95*len(warm_rtt))]:8.2f} ms")
    print(f"  compute  avg={statistics.mean(warm_srv):8.2f} ms")
    print(f"  NETWORK  avg={statistics.mean(net):8.2f} ms   <-- SMART_NETWORK_MS")
    print(f"  compute  avg={statistics.mean(warm_srv):8.2f} ms   <-- SMART_COMPUTE_MS")


if __name__ == "__main__":
    main()
