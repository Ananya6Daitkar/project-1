# iot_simulator.py
#
# simulates a satellite-to-ground IoT pipeline in pure software.
# the idea is to model what a real system would look like:
#
#   satellite (edge node)  -->  MQTT  -->  ground station
#
# no real broker needed — everything runs in-process.
# to swap in a real Mosquitto broker later, just replace InProcessBroker
# with paho.mqtt.client — the interface is the same.
#
# Usage:
#   python src/iot_simulator.py --live            <- continuous stream
#   python src/iot_simulator.py --burst 20        <- emit 20 events and stop
#   python src/iot_simulator.py --target Mars

import json
import time
import random
import threading
import argparse
import datetime
from pathlib import Path
from typing import Optional, List, Dict, Callable
from dataclasses import dataclass, asdict

import logging
logger = logging.getLogger("iot")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")

# which planets we can "orbit"
PLANETARY_TARGETS = ["Moon", "Mars", "Mercury"]

# rough orbital parameters for each planet
ORBITAL_PARAMS = {
    "Moon":    {"altitude_km": (50,   100), "velocity_km_s": (1.5, 1.8)},
    "Mars":    {"altitude_km": (250,  400), "velocity_km_s": (3.3, 3.5)},
    "Mercury": {"altitude_km": (200,  500), "velocity_km_s": (2.9, 3.1)},
}

INSTRUMENTS = ["NAC-Camera", "WAC-Camera", "HiRISE-IR", "CTX-Imager"]

# MQTT topic names
TOPIC_DETECTIONS = "crater/detections"
TOPIC_HEALTH     = "crater/health"
TOPIC_ALERTS     = "crater/alerts"
TOPIC_TELEMETRY  = "crater/telemetry"


# ----------------------------------------------------------------
# sensor data structure
# ----------------------------------------------------------------

@dataclass
class SensorPayload:
    """one packet of data the satellite sends down after processing a tile"""
    # who/what
    satellite_id:   str
    orbit_number:   int
    target_body:    str
    instrument:     str
    timestamp_utc:  str

    # where we are
    altitude_km:    float
    velocity_km_s:  float
    latitude_deg:   float
    longitude_deg:  float
    solar_angle_deg: float

    # instrument health
    sensor_temp_c:    float
    power_draw_w:     float
    storage_used_pct: float

    # what the model found
    n_craters_detected: int
    crater_classes:     List[str]
    avg_confidence:     float
    inference_ms:       float
    tile_id:            str

    # flags the edge node computes before downlinking
    high_density_flag: bool = False
    landing_safe_flag: bool = True
    anomaly_flag:      bool = False

    def to_json(self):
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(s):
        return SensorPayload(**json.loads(s))


def generate_sensor_payload(n_craters, crater_classes, avg_conf,
                             inference_ms, tile_id="tile_0000", target=None):
    """fills out a SensorPayload with realistic random values"""
    target = target or random.choice(PLANETARY_TARGETS)
    params = ORBITAL_PARAMS[target]
    now    = datetime.datetime.utcnow().isoformat() + "Z"

    altitude  = round(random.uniform(*params["altitude_km"]), 2)
    velocity  = round(random.uniform(*params["velocity_km_s"]), 3)
    lat       = round(random.uniform(-90, 90), 4)
    lon       = round(random.uniform(-180, 180), 4)
    temp_c    = round(random.uniform(-40, 120), 1)  # space is harsh
    power_w   = round(random.uniform(8, 25), 2)
    storage   = round(random.uniform(10, 95), 1)

    return SensorPayload(
        satellite_id="LRO-MOCK",
        orbit_number=random.randint(1000, 9999),
        target_body=target,
        instrument=random.choice(INSTRUMENTS),
        timestamp_utc=now,
        altitude_km=altitude,
        velocity_km_s=velocity,
        latitude_deg=lat,
        longitude_deg=lon,
        solar_angle_deg=round(random.uniform(0, 90), 1),
        sensor_temp_c=temp_c,
        power_draw_w=power_w,
        storage_used_pct=storage,
        n_craters_detected=n_craters,
        crater_classes=crater_classes,
        avg_confidence=round(avg_conf, 3),
        inference_ms=round(inference_ms, 2),
        tile_id=tile_id,
        high_density_flag=n_craters > 12,
        landing_safe_flag=n_craters < 6 and avg_conf < 0.85,
        anomaly_flag=temp_c > 110 or power_w > 23,  # instrument getting hot or power spike
    )


# ----------------------------------------------------------------
# in-process MQTT broker — no external server needed
# ----------------------------------------------------------------

class InProcessBroker:
    """
    a dead-simple pub/sub message bus that runs in memory.
    works exactly like MQTT: you publish to a topic, subscribers get called.
    drop-in replaceable with paho.mqtt if you want a real broker later.
    """
    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {}
        self._message_log: List[Dict] = []

    def subscribe(self, topic, callback):
        self._subscribers.setdefault(topic, []).append(callback)

    def publish(self, topic, payload, retain=False):
        msg = {
            "topic":     topic,
            "payload":   payload,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "retained":  retain,
        }
        self._message_log.append(msg)
        for cb in self._subscribers.get(topic, []):
            try:
                cb(topic, payload)
            except Exception as e:
                logger.warning(f"subscriber error on {topic}: {e}")

    def get_log(self):
        return self._message_log.copy()


# shared broker instance
_broker = InProcessBroker()


# ----------------------------------------------------------------
# edge node — simulates the satellite's on-board processor
# ----------------------------------------------------------------

class EdgeNode:
    """
    pretends to be the camera + inference chip on the satellite.
    in the real world this would run on an ARM or FPGA board in orbit.
    here it just generates fake detections and publishes them.
    """
    def __init__(self, broker, target="Moon", interval_sec=2.0):
        self.broker        = broker
        self.target        = target
        self.interval_sec  = interval_sec
        self._running      = False
        self._thread       = None
        self._tile_counter = 0

    def _simulate_detection(self):
        """make up a plausible detection result for one tile"""
        n_craters    = random.randint(0, 20)
        classes      = random.choices(
            ["small_crater", "medium_crater", "large_crater"],
            weights=[0.60, 0.30, 0.10], k=n_craters
        )
        tile_id      = f"tile_{self._tile_counter:06d}"
        self._tile_counter += 1
        return generate_sensor_payload(
            n_craters=n_craters,
            crater_classes=classes,
            avg_conf=round(random.uniform(0.30, 0.95), 3),
            inference_ms=round(random.uniform(5, 30), 2),
            tile_id=tile_id,
            target=self.target,
        )

    def _publish_event(self, payload):
        """sends the detection + health + any alerts to the broker"""
        self.broker.publish(TOPIC_DETECTIONS, payload.to_json())

        # regular health ping
        self.broker.publish(TOPIC_HEALTH, json.dumps({
            "satellite_id": payload.satellite_id,
            "temp_c":       payload.sensor_temp_c,
            "power_w":      payload.power_draw_w,
            "storage_pct":  payload.storage_used_pct,
            "timestamp":    payload.timestamp_utc,
        }))

        # fire alerts if needed
        if payload.high_density_flag:
            self.broker.publish(TOPIC_ALERTS, json.dumps({
                "type":    "HIGH_CRATER_DENSITY",
                "tile_id": payload.tile_id,
                "count":   payload.n_craters_detected,
                "lat":     payload.latitude_deg,
                "lon":     payload.longitude_deg,
                "ts":      payload.timestamp_utc,
            }))

        if payload.anomaly_flag:
            self.broker.publish(TOPIC_ALERTS, json.dumps({
                "type":    "INSTRUMENT_ANOMALY",
                "temp_c":  payload.sensor_temp_c,
                "power_w": payload.power_draw_w,
                "ts":      payload.timestamp_utc,
            }))

    def emit_once(self):
        """fire one event immediately"""
        p = self._simulate_detection()
        self._publish_event(p)
        return p

    def start_stream(self):
        """kick off a background thread that keeps sending events"""
        self._running = True
        def _loop():
            logger.info(f"[EdgeNode] streaming to {self.target} every {self.interval_sec}s")
            while self._running:
                p = self.emit_once()
                logger.debug(f"[EdgeNode] tile={p.tile_id} craters={p.n_craters_detected} conf={p.avg_confidence:.2f}")
                time.sleep(self.interval_sec)
        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop_stream(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("[EdgeNode] stopped")


# ----------------------------------------------------------------
# ground station — receives and stores everything
# ----------------------------------------------------------------

class GroundStation:
    """
    listens to the broker and collects all incoming events.
    also keeps a "device shadow" — the latest known state of the satellite,
    which is the IoT equivalent of a digital twin.
    """
    def __init__(self, broker):
        self.broker  = broker
        self._events  = []
        self._alerts  = []
        self._health  = []
        self._lock    = threading.Lock()

        broker.subscribe(TOPIC_DETECTIONS, self._on_detection)
        broker.subscribe(TOPIC_ALERTS,     self._on_alert)
        broker.subscribe(TOPIC_HEALTH,     self._on_health)

    def _on_detection(self, topic, payload):
        try:
            event = SensorPayload.from_json(payload)
            with self._lock:
                self._events.append(event)
        except Exception as e:
            logger.warning(f"[GroundStation] parse error: {e}")

    def _on_alert(self, topic, payload):
        with self._lock:
            self._alerts.append(json.loads(payload))

    def _on_health(self, topic, payload):
        with self._lock:
            self._health.append(json.loads(payload))

    @property
    def device_shadow(self):
        """latest known state of the satellite — like a digital twin snapshot"""
        with self._lock:
            if not self._events:
                return None
            latest = self._events[-1]
            return {
                "satellite_id":           latest.satellite_id,
                "target_body":            latest.target_body,
                "altitude_km":            latest.altitude_km,
                "sensor_temp_c":          latest.sensor_temp_c,
                "power_draw_w":           latest.power_draw_w,
                "storage_used_pct":       latest.storage_used_pct,
                "last_seen":              latest.timestamp_utc,
                "total_craters_detected": sum(e.n_craters_detected for e in self._events),
                "total_tiles_processed":  len(self._events),
                "active_alerts":          len(self._alerts),
            }

    def get_events(self):
        with self._lock:
            return self._events.copy()

    def get_alerts(self):
        with self._lock:
            return self._alerts.copy()

    def get_telemetry_series(self):
        """returns lists of values over time — useful for dashboard charts"""
        with self._lock:
            return {
                "timestamps":    [e.timestamp_utc for e in self._events],
                "crater_counts": [e.n_craters_detected for e in self._events],
                "confidences":   [e.avg_confidence for e in self._events],
                "altitudes_km":  [e.altitude_km for e in self._events],
                "temps_c":       [e.sensor_temp_c for e in self._events],
            }

    def print_dashboard(self):
        shadow = self.device_shadow
        if not shadow:
            logger.info("no data yet")
            return
        logger.info("\n=== ground station ===")
        for k, v in shadow.items():
            logger.info(f"  {k:<28}: {v}")
        for a in self._alerts[-3:]:
            logger.warning(f"  ⚠ {a['type']} — {a}")
        logger.info("=" * 40)


# ----------------------------------------------------------------
# telemetry file logger
# ----------------------------------------------------------------

class TelemetryLogger:
    """writes every message to a .jsonl file so you can replay or analyse it later"""
    def __init__(self, log_path):
        self.log_path = log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)

    def subscribe(self, broker):
        for topic in [TOPIC_DETECTIONS, TOPIC_HEALTH, TOPIC_ALERTS]:
            broker.subscribe(topic, self._write)

    def _write(self, topic, payload):
        entry = {"topic": topic, "payload": payload,
                 "ts": datetime.datetime.utcnow().isoformat()}
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")


# ----------------------------------------------------------------
# public API — called from app.py and iot_dashboard.py
# ----------------------------------------------------------------

_ground_station: Optional[GroundStation] = None
_edge_node:      Optional[EdgeNode]      = None


def start_iot_simulation(target="Moon", interval_sec=2.0):
    """starts the full IoT stack — safe to call multiple times"""
    global _ground_station, _edge_node, _broker
    if _edge_node and _edge_node._running:
        return _ground_station

    _broker         = InProcessBroker()
    _ground_station = GroundStation(_broker)

    log_path = Path(__file__).resolve().parent.parent / "data" / "telemetry.jsonl"
    TelemetryLogger(log_path).subscribe(_broker)

    _edge_node = EdgeNode(_broker, target=target, interval_sec=interval_sec)
    _edge_node.start_stream()
    return _ground_station


def stop_iot_simulation():
    global _edge_node
    if _edge_node:
        _edge_node.stop_stream()


def get_ground_station():
    return _ground_station


def emit_detection_event(n_craters, crater_classes, avg_conf,
                          inference_ms, tile_id="manual", target="Moon"):
    """
    called from app.py after a real inference — sends the result
    into the IoT pipeline so it shows up on the dashboard
    """
    global _ground_station, _broker
    if _ground_station is None:
        _broker         = InProcessBroker()
        _ground_station = GroundStation(_broker)

    node    = EdgeNode(_broker, target=target)
    payload = generate_sensor_payload(n_craters, crater_classes, avg_conf,
                                      inference_ms, tile_id, target)
    node._publish_event(payload)
    return payload


# ----------------------------------------------------------------
# CLI
# ----------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IoT telemetry simulator")
    parser.add_argument("--live",     action="store_true", help="stream forever (Ctrl-C to stop)")
    parser.add_argument("--burst",    type=int, default=0,  help="emit N events then exit")
    parser.add_argument("--target",   type=str, default="Moon", choices=PLANETARY_TARGETS)
    parser.add_argument("--interval", type=float, default=1.5)
    args = parser.parse_args()

    broker  = InProcessBroker()
    station = GroundStation(broker)
    node    = EdgeNode(broker, target=args.target, interval_sec=args.interval)

    if args.burst > 0:
        logger.info(f"emitting {args.burst} events...")
        for i in range(args.burst):
            p = node.emit_once()
            logger.info(f"[{i+1}/{args.burst}] tile={p.tile_id}  "
                        f"craters={p.n_craters_detected}  conf={p.avg_confidence:.2f}  "
                        f"alt={p.altitude_km}km")
        station.print_dashboard()

    elif args.live:
        node.start_stream()
        logger.info("streaming... Ctrl-C to stop")
        try:
            while True:
                time.sleep(5)
                station.print_dashboard()
        except KeyboardInterrupt:
            node.stop_stream()
            logger.info("stopped")
    else:
        parser.print_help()
