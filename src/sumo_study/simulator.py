"""SUMO pipeline: network, activitygen demand, routing, simulation, output parsing and caching.

One simulation = ``activitygen -> duarouter -> sumo`` for one parameter vector and one seed.
Successful runs are cached under ``runs/<key>/result.json``. The key hashes the simulator
fingerprint (SUMO version, this file, physics, network), the parameters and the seed, so a
cached result can only ever be reused for an identical simulation.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from .config import BOUNDS, FEATURES, PHYSICS
from .io import atomic_json, save_csv, stable_hash

EXECUTABLES = ["sumo", "netgenerate", "activitygen", "duarouter"]


def _checked(cmd, cwd: Path, logname: str, timeout: int = 3600) -> float:
    start = time.monotonic()
    with open(cwd / logname, "w") as log:
        result = subprocess.run([str(c) for c in cmd], cwd=cwd, stdout=log,
                                stderr=subprocess.STDOUT, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}); see {cwd / logname}")
    return time.monotonic() - start


def sumo_version() -> str:
    for exe in EXECUTABLES:
        if shutil.which(exe) is None:
            raise RuntimeError(f"Missing SUMO executable '{exe}'. Install Eclipse SUMO 1.28.0.")
    return subprocess.check_output(["sumo", "--version"], text=True).splitlines()[0]


class Simulator:
    """A fixed synthetic grid network with the activity-based demand generator attached.

    Parameters
    ----------
    root : folder holding the network and the run cache.
    grid_number : junctions per side (6 -> 120 directed edges, 8 -> 224).
    teleport : SUMO ``--time-to-teleport`` in s (-1 disables teleporting).
    """

    def __init__(self, root, grid_number=PHYSICS["grid_number"], teleport=PHYSICS["teleport"],
                 workers=4, retain_raw=True, extra_sumo_args=()):
        self.physics = dict(PHYSICS, grid_number=int(grid_number), teleport=int(teleport))
        self.extra_sumo_args = [str(a) for a in extra_sumo_args]
        self.workers = max(1, int(workers))
        self.retain_raw = retain_raw
        tag = f"grid{grid_number}_tp{teleport}" + ("_" + stable_hash(self.extra_sumo_args)[:6]
                                                    if self.extra_sumo_args else "")
        self.root = Path(root).resolve() / tag
        (self.root / "runs").mkdir(parents=True, exist_ok=True)
        self.net_file = self.root / "network.net.xml"
        if not self.net_file.exists():
            p = self.physics
            _checked(["netgenerate", "--grid", "--grid.number", p["grid_number"],
                      "--grid.length", p["block_length"], "--default.lanenumber", p["lanes"],
                      "--default.speed", p["speed"], "--tls.guess", "true",
                      "--output-file", self.net_file], self.root, "netgenerate.log")
        self._zone_network()
        self.version = sumo_version()
        self.fingerprint = stable_hash(dict(
            sumo=self.version, physics=self.physics, extra=self.extra_sumo_args,
            code=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            network=hashlib.sha256(self.net_file.read_bytes()).hexdigest()))
        atomic_json(self.root / "simulator.json", dict(
            sumo=self.version, physics=self.physics, extra_sumo_args=self.extra_sumo_args,
            fingerprint=self.fingerprint, edges=len(self.edges),
            zones=dict(employment=len(self.cbd_edges), residential=len(self.res_edges)),
            gates=self.gate_edges, schools=self.school_edges,
            bus_forward=self.bus_fwd, bus_reverse=self.bus_rev))

    # ---- network zoning (deterministic, edge-order based; documented as a limitation) ----
    def _zone_network(self):
        net = ET.parse(self.net_file).getroot()
        self.edges = [e.attrib["id"] for e in net.findall("edge")
                      if "function" not in e.attrib and not e.attrib["id"].startswith(":")]
        n = len(self.edges)
        self.cbd_edges = self.edges[int(n * .35):int(n * .65)]
        self.res_edges = self.edges[:int(n * .35)] + self.edges[int(n * .65):]
        self.gate_edges = [self.edges[0], self.edges[5], self.edges[-6], self.edges[-1]]
        self.school_edges = [self.edges[10], self.edges[n // 2], self.edges[-10]]
        self.bus_fwd = self.edges[::8][:6]
        self.bus_rev = self.edges[1::8][:6]

    def geometry(self) -> dict:
        net = ET.parse(self.net_file).getroot()
        out = {}
        for edge in net.findall("edge"):
            eid = edge.attrib["id"]
            if eid not in self.edges:
                continue
            lane = edge.find("lane")
            pts = [[float(v) for v in xy.split(",")] for xy in lane.attrib["shape"].split()]
            zone = "employment" if eid in self.cbd_edges else "residential"
            out[eid] = dict(points=pts, length=float(lane.attrib["length"]), zone=zone,
                            bus=eid in self.bus_fwd + self.bus_rev,
                            gate=eid in self.gate_edges, school=eid in self.school_edges)
        return out

    # ---- activitygen statistics file ----
    def write_stat_xml(self, params: dict, path: Path):
        p = dict(params, inhabitants=self.physics["inhabitants"],
                 incomingTraffic=self.physics["incomingTraffic"])
        city = ET.Element("city")
        ET.SubElement(city, "general", {
            "inhabitants": str(int(p["inhabitants"])),
            "households": str(int(p["inhabitants"] * 0.45)),
            "childrenAgeLimit": "19", "retirementAgeLimit": "65",
            "carRate": f"{p['carRate']:.4f}", "unemploymentRate": f"{p['unemploymentRate']:.4f}",
            "footDistanceLimit": f"{p['footDistanceLimit']:.1f}",
            "incomingTraffic": str(int(p["incomingTraffic"])),
            "outgoingTraffic": str(int(p["inhabitants"] * 0.08))})
        ET.SubElement(city, "parameters", {
            "carPreference": f"{p['carPreference']:.4f}", "meanTimePerKmInCity": "180",
            "freeTimeActivityRate": f"{p['freeTimeActivityRate']:.4f}",
            "uniformRandomTraffic": f"{p['uniformRandomTraffic']:.4f}",
            "departureVariation": f"{p['departureVariation']:.1f}"})
        pop = ET.SubElement(city, "population")
        for a, b, n in [(0, 19, 220), (19, 40, 380), (40, 65, 300), (65, 90, 100)]:
            ET.SubElement(pop, "bracket", {"beginAge": str(a), "endAge": str(b), "peopleNbr": str(n)})
        wh = ET.SubElement(city, "workHours")
        peak = p["morningPeakProp"]
        ET.SubElement(wh, "opening", {"hour": "28800", "proportion": f"{peak:.3f}"})
        ET.SubElement(wh, "opening", {"hour": "32400", "proportion": f"{1 - peak:.3f}"})
        ET.SubElement(wh, "closing", {"hour": "61200", "proportion": "0.50"})
        ET.SubElement(wh, "closing", {"hour": "64800", "proportion": "0.50"})
        streets = ET.SubElement(city, "streets")
        for eid in self.edges:
            ET.SubElement(streets, "street", {
                "edge": eid, "population": "25.0" if eid in self.res_edges else "5.0",
                "workPosition": "40.0" if eid in self.cbd_edges else "4.0"})
        gates = ET.SubElement(city, "cityGates")
        for eid in self.gate_edges:
            ET.SubElement(gates, "entrance", {"edge": eid, "pos": "10.0",
                                              "incoming": "0.25", "outgoing": "0.25"})
        schools = ET.SubElement(city, "schools")
        for eid in self.school_edges:
            ET.SubElement(schools, "school", {"edge": eid, "pos": "100.0", "beginAge": "6",
                                              "endAge": "18", "capacity": "350",
                                              "opening": "28800", "closing": "54000"})
        stations = ET.SubElement(city, "busStations")
        stops = self.bus_fwd + self.bus_rev
        for i, eid in enumerate(stops):
            ET.SubElement(stations, "busStation", {"id": str(i + 1), "edge": eid, "pos": "50.0"})
        lines = ET.SubElement(city, "busLines")
        line = ET.SubElement(lines, "busLine", {"id": "101", "maxTripDuration": "1800"})
        fwd = ET.SubElement(line, "stations")
        for i in range(len(self.bus_fwd)):
            ET.SubElement(fwd, "station", {"refId": str(i + 1)})
        rev = ET.SubElement(line, "revStations")
        for i in range(len(self.bus_fwd), len(stops)):
            ET.SubElement(rev, "station", {"refId": str(i + 1)})
        freq = ET.SubElement(line, "frequencies")
        ET.SubElement(freq, "frequency", {"begin": "21600", "end": "72000",
                                          "rate": str(int(p["busHeadway"]))})
        tree = ET.ElementTree(city)
        ET.indent(tree, space="  ")
        tree.write(path, encoding="utf-8", xml_declaration=True)

    # ---- output parsing ----
    def _parse(self, d: Path) -> dict:
        begin, end = self.physics["begin"], self.physics["end"]
        trips = [t.attrib for t in ET.parse(d / "tripinfo.xml").getroot().iter("tripinfo")]
        df = pd.DataFrame(dict(
            id=[a["id"] for a in trips],
            depart=[float(a.get("depart", -1)) for a in trips],
            arrival=[float(a.get("arrival", -1)) for a in trips],
            timeLoss=[float(a.get("timeLoss", 0)) for a in trips],
            routeLength=[float(a.get("routeLength", 0)) for a in trips],
            waitingTime=[float(a.get("waitingTime", 0)) for a in trips],
            departDelay=[float(a.get("departDelay", 0)) for a in trips],
            vtype=[a.get("vType", "unknown") for a in trips],
            vaporized=[a.get("vaporized", "") for a in trips]))
        if df.empty:
            raise RuntimeError(f"No trip records in {d}")
        done = df[(df.arrival >= 0) & (df.depart >= 0) & (~df.vaporized.isin(["true", "1"]))]
        if done.empty:
            raise RuntimeError(f"No completed trips in {d}")
        r = dict(
            mean_time_loss_sec=float(done.timeLoss.mean()),
            total_vkt_km=float(done.routeLength.sum() / 1000),
            total_completed_trips=int(len(done)),
            unfinished_trips=int(((df.arrival < 0) & (df.depart >= 0)).sum()),
            recorded_trips=int(len(df)),
            completion_fraction_recorded=float(len(done) / len(df)),
            mean_depart_delay_sec=float(done.departDelay.mean()),
            mean_waiting_sec=float(done.waitingTime.mean()),
            mean_time_loss_started_sec=float(df[df.depart >= 0].timeLoss.mean()),
            p90_time_loss_sec=float(done.timeLoss.quantile(.9)))
        # Buses accrue dwell time at stops as SUMO timeLoss; report general traffic separately.
        gen = done[done.vtype != "bus"]
        r["general_time_loss_sec"] = float(gen.timeLoss.mean())
        r["general_vkt_km"] = float(gen.routeLength.sum() / 1000)
        stat = ET.parse(d / "statistics.xml").getroot()
        for el in ["vehicles", "teleports", "safety", "vehicleTripStatistics"]:
            node = stat.find(".//" + el)
            if node is not None:
                for k, v in node.attrib.items():
                    try:
                        r[f"stat_{el}_{k}"] = float(v)
                    except ValueError:
                        pass
        # Completed trips and delay per vehicle type: default = activity cars,
        # random = uniform random traffic, bus = scheduled bus vehicles (not passengers).
        for vt in ["default", "random", "bus"]:
            g = done[done.vtype == vt]
            r[f"type_{vt}_completed"] = int(len(g))
            r[f"type_{vt}_delay"] = float(g.timeLoss.mean()) if len(g) else float("nan")
            r[f"type_{vt}_vkt_km"] = float(g.routeLength.sum() / 1000)
        # Generated demand inside the simulated window, by type (explains mechanisms).
        routes = ET.parse(d / "routes.rou.xml").getroot()
        counts = {"default": 0, "random": 0, "bus": 0}
        for v in routes:
            if v.tag in ("vehicle", "trip") and begin <= float(v.attrib.get("depart", -1)) < end:
                t = v.attrib.get("type", "default")
                counts[t] = counts.get(t, 0) + 1
        for t, c in counts.items():
            r[f"demand_{t}_in_window"] = int(c)
        inserted = r.get("stat_vehicles_inserted", np.nan)
        r["served_fraction"] = float(len(done) / inserted) if inserted else float("nan")
        hours = np.floor(done.arrival / 3600).astype(int)
        hourly = hours.value_counts().reindex(range(begin // 3600, math.ceil(end / 3600)),
                                              fill_value=0)
        for h, c in hourly.items():
            r[f"arrivals_hour_{h}"] = int(c)
        return r

    # ---- one simulation ----
    def key(self, params: dict, seed: int) -> str:
        return stable_hash(dict(fingerprint=self.fingerprint, params=params, seed=int(seed)))[:24]

    def simulate(self, params: dict, seed: int) -> dict:
        params = {k: float(params[k]) for k in FEATURES}
        for k, v in params.items():
            lo, hi = BOUNDS[k]
            if not lo - 1e-9 <= v <= hi + 1e-9:
                raise ValueError(f"{k}={v} outside [{lo}, {hi}]")
        key = self.key(params, seed)
        d = self.root / "runs" / key
        cached = d / "result.json"
        if cached.exists():
            return json.loads(cached.read_text())
        d.mkdir(exist_ok=True)
        atomic_json(d / "run_config.json", dict(params=params, seed=int(seed),
                                                fingerprint=self.fingerprint))
        self.write_stat_xml(params, d / "demand.stat.xml")
        p = self.physics
        cmds = [
            ["activitygen", "--net-file", self.net_file, "--stat-file", d / "demand.stat.xml",
             "--output-file", d / "trips.rou.xml", "--seed", seed],
            ["duarouter", "--net-file", self.net_file, "--route-files", d / "trips.rou.xml",
             "--output-file", d / "routes.rou.xml", "--seed", seed],
            ["sumo", "--net-file", self.net_file, "--route-files", d / "routes.rou.xml",
             "--begin", p["begin"], "--end", p["end"], "--time-to-teleport", p["teleport"],
             "--tripinfo-output", d / "tripinfo.xml", "--tripinfo-output.write-unfinished", "true",
             "--statistic-output", d / "statistics.xml", "--no-step-log", "true",
             "--seed", seed, *self.extra_sumo_args]]
        try:
            elapsed = sum(_checked(c, d, f"stage_{i}.log") for i, c in enumerate(cmds))
            result = dict(self._parse(d), **params, seed=int(seed), run_key=key,
                          elapsed_sec=elapsed)
            for f in ["trips.rou.xml", "routes.rou.xml", "tripinfo.xml", "statistics.xml"]:
                path = d / f
                if self.retain_raw and f != "trips.rou.xml":
                    with open(path, "rb") as src, gzip.open(f"{path}.gz", "wb") as dst:
                        shutil.copyfileobj(src, dst)
                path.unlink(missing_ok=True)
            (d / "routes.rou.alt.xml").unlink(missing_ok=True)
            atomic_json(cached, result)
            (d / "failure.txt").unlink(missing_ok=True)
            return result
        except Exception as exc:  # recorded, never silently dropped
            (d / "failure.txt").write_text(repr(exc))
            raise

    # ---- many simulations ----
    def run(self, jobs: list[dict], name: str, table_dir: Path | None = None) -> pd.DataFrame:
        """Run jobs ``{params, seed, meta}`` in parallel. Any failure stops the analysis."""
        results, failures = [], []
        start = time.monotonic()
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {pool.submit(self.simulate, j["params"], j["seed"]): j for j in jobs}
            for i, fut in enumerate(as_completed(futures), 1):
                job = futures[fut]
                try:
                    results.append(dict(fut.result(), **job.get("meta", {})))
                except Exception as exc:
                    failures.append(dict(seed=job["seed"], params=job["params"], error=repr(exc)))
                if i % max(1, len(jobs) // 10) == 0 or i == len(jobs):
                    print(f"  {name}: {i}/{len(jobs)} runs, {len(failures)} failures, "
                          f"{(time.monotonic() - start) / 60:.1f} min", flush=True)
        if table_dir is not None:
            atomic_json(Path(table_dir) / f"{name}_failures.json", failures)
        if failures:
            raise RuntimeError(f"{len(failures)} simulations failed in '{name}'. Successful runs "
                               "are cached; inspect runs/<key>/stage_*.log and rerun.")
        sort = [c for c in ["setting_id", "replicate", "seed"] if c in results[0]]
        df = pd.DataFrame(results).sort_values(sort).reset_index(drop=True)
        if table_dir is not None:
            save_csv(df, Path(table_dir) / f"{name}.csv")
        return df
