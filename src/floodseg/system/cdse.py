"""Copernicus Data Space Ecosystem: catalog lookup and OData download of whole S2 L1C products.

The Process API is not used here: a whole tile through it costs about 2 000 processing units
(docs/system_evaluation_protocol.md §1). OData downloads cost none and accept the same
client-credentials token as Sentinel Hub (checked 2026-10-18).
"""

import os
from collections.abc import Callable
from pathlib import Path

import requests

from floodseg.target_pool import REPO

TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
CATALOG = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
DOWNLOAD = "https://download.dataspace.copernicus.eu/odata/v1/Products({id})/$value"
CHUNK = 8 << 20


def _env(name: str) -> str:
    env = REPO / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith(f"{name}="):
                return line.split("=", 1)[1].strip()
    return os.environ[name]


def token() -> str:
    r = requests.post(TOKEN_URL, data={
        "grant_type": "client_credentials",
        "client_id": _env("CDSE_CLIENT_ID"),
        "client_secret": _env("CDSE_CLIENT_SECRET"),
    }, timeout=30)
    r.raise_for_status()
    return r.json()["access_token"]


def find_l1c(tile: str, date: str) -> list[dict]:
    """L1C products of one MGRS tile (e.g. "17MNP") acquired on one date (YYYY-MM-DD), newest
    processing first. Each item: Id, Name, ContentLength (bytes)."""
    f = (f"Collection/Name eq 'SENTINEL-2' and contains(Name,'MSIL1C') and contains(Name,'_T{tile}_') "
         f"and ContentDate/Start ge {date}T00:00:00.000Z and ContentDate/Start le {date}T23:59:59.999Z")
    r = requests.get(CATALOG, params={"$filter": f, "$top": 20, "$orderby": "PublicationDate desc"}, timeout=60)
    r.raise_for_status()
    return [{k: p[k] for k in ("Id", "Name", "ContentLength")} for p in r.json()["value"] if p.get("Online", True)]


def acquisitions(tile: str, start: str, end: str) -> list[dict]:
    """L1C products of one tile between two dates (inclusive), oldest first, with cloud cover."""
    f = (f"Collection/Name eq 'SENTINEL-2' and contains(Name,'MSIL1C') and contains(Name,'_T{tile}_') "
         f"and ContentDate/Start ge {start}T00:00:00.000Z and ContentDate/Start le {end}T23:59:59.999Z")
    r = requests.get(CATALOG, params={"$filter": f, "$top": 100, "$orderby": "ContentDate/Start asc",
                                      "$expand": "Attributes"}, timeout=60)
    r.raise_for_status()
    out = []
    for p in r.json()["value"]:
        attrs = {a["Name"]: a.get("Value") for a in p.get("Attributes", [])}
        out.append({"Id": p["Id"], "Name": p["Name"], "date": p["ContentDate"]["Start"][:10],
                    "cloud_cover": attrs.get("cloudCover"), "ContentLength": p["ContentLength"]})
    return out


def download(product: dict, dest: Path, tok: str, progress: Callable[[int, int], None] | None = None) -> Path:
    """Stream one product archive to `dest` (a .zip path). `progress(done_bytes, total_bytes)`."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(".part")
    with requests.Session() as s:
        s.headers["Authorization"] = f"Bearer {tok}"
        url = DOWNLOAD.format(id=product["Id"])
        r = s.get(url, stream=True, allow_redirects=False, timeout=120)
        while r.status_code in (301, 302, 303, 307, 308):  # keep the token across redirects
            url = r.headers["Location"]
            r = s.get(url, stream=True, allow_redirects=False, timeout=120)
        r.raise_for_status()
        total = int(r.headers.get("Content-Length") or product.get("ContentLength") or 0)
        done = 0
        with open(part, "wb") as f:
            for chunk in r.iter_content(CHUNK):
                f.write(chunk)
                done += len(chunk)
                if progress:
                    progress(done, total)
    if total and done != total:
        raise IOError(f"incomplete download: {done} of {total} bytes")
    part.rename(dest)
    return dest
