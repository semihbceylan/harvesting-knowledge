"""
download_images_all_years.py  — version 3

• Walks every TARLA‑DISK directory on the server.
• Processes only the years ≥ 2013 (MIN_YEAR = 2013). Any disk/year folder
  earlier than this is skipped, except for TARLA-DISK 2014, where only 2013 and 2014 are allowed.
• Keeps the same local structure and corruption-handling logic.
"""

import os
import re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import paramiko
from PIL import Image
from dotenv import load_dotenv
import shutil

###############################################################################
# Configuration
###############################################################################
MIN_YEAR        = 2013
TARGET_TIME     = "10_00"
TIME_START      = "08_00"
TIME_END        = "12_00"
ZOOM_OPTIONS    = ["1X", "10X"]
MAX_WORKERS     = 4
LOCAL_ROOT      = "F:/test"
CORRUPTED_DIR   = os.path.join(LOCAL_ROOT, "corrupted")

###############################################################################
# Environment / connection
###############################################################################
load_dotenv()
hostname = os.getenv("HOSTNAME")
port     = int(os.getenv("PORT", "22"))
username = os.getenv("UNAME")
password = os.getenv("PASSWORD")
print(f"Connecting to {hostname}:{port} as {username}…")

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(hostname, port=port, username=username, password=password)
sftp = client.open_sftp()

os.makedirs(CORRUPTED_DIR, exist_ok=True)
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

###############################################################################
# Helpers
###############################################################################

def extract_time(filename: str) -> datetime:
    m = re.search(r"(\d{4}-\d{2}-\d{2})-(\d{2}_\d{2})", filename)
    if not m:
        raise ValueError(f"Filename does not match expected pattern: {filename}")
    return datetime.strptime(f"{m.group(1)} {m.group(2)}", "%Y-%m-%d %H_%M")


def validate_and_quarantine(path: str) -> None:
    try:
        with Image.open(path) as img:
            img.verify()
    except Exception:
        print(f"[CORRUPT] {path}")
        try:
            shutil.move(path, os.path.join(CORRUPTED_DIR, os.path.basename(path)))
        except Exception as exc:
            print(f"[ERROR] Failed to quarantine: {exc}")


def should_download(local_dir: str, day_key: str, option: str) -> bool:
    if not os.path.exists(local_dir):
        return True
    pattern = re.compile(fr"^{day_key}-\d{{2}}_\d{{2}}-{option.lower()}\.jpeg$")
    return not any(pattern.match(f) for f in os.listdir(local_dir) if f.lower().endswith(".jpeg"))

###############################################################################
# Walk remote tree
###############################################################################
ROOT_REMOTE = "/share/TARBIL"
special_2014_years = {"2013", "2014"}

disk_dirs = [
    d for d in sftp.listdir(ROOT_REMOTE)
    if d.startswith("TARLA-DISK")
]

for disk in sorted(disk_dirs):
    disk_path = f"{ROOT_REMOTE}/{disk}"
    try:
        all_years = sftp.listdir(disk_path)
    except Exception:
        continue

    if disk == "TARLA-DISK 2014":
        year_dirs = [y for y in all_years if y in special_2014_years]
    else:
        year_dirs = [y for y in all_years if re.fullmatch(r"\d{4}", y) and int(y) >= MIN_YEAR]

    if not year_dirs:
        continue

    print(f"▶ Disk {disk}  (years: {', '.join(year_dirs)})")

    for year in sorted(year_dirs):
        year_path = f"{disk_path}/{year}"
        try:
            stations = sftp.listdir(year_path)
        except Exception:
            continue

        for station in sorted(stations):
            if not re.fullmatch(r"\d{2}\.\d{2}", station):
                continue

            station_path = f"{year_path}/{station}"
            try:
                cameras = sftp.listdir(station_path)
            except Exception:
                continue

            for camera in sorted(cameras):
                if not camera.startswith("K"):
                    continue

                camera_year_path = f"{station_path}/{camera}/{year}"
                try:
                    months = sftp.listdir(camera_year_path)
                except Exception:
                    continue

                for month in sorted(m for m in months if m.isdigit() and len(m) == 2):
                    month_path = f"{camera_year_path}/{month}"
                    try:
                        files = sftp.listdir(month_path)
                    except Exception:
                        continue

                    files.sort()
                    for day in range(1, 32):
                        try:
                            day_dt = datetime.strptime(f"{year}-{month}-{day:02d} {TARGET_TIME}", "%Y-%m-%d %H_%M")
                        except ValueError:
                            continue

                        day_key = day_dt.strftime("%Y_%m_%d")
                        for option in ZOOM_OPTIONS:
                            local_dir = os.path.join(LOCAL_ROOT, station, year, camera, option)
                            if not should_download(local_dir, day_key, option):
                                continue

                            filtered = [f for f in files if option in f]
                            if not filtered:
                                continue

                            closest_file = None
                            closest_diff = float("inf")
                            exact_found = False

                            for fname in filtered:
                                try:
                                    dt = extract_time(fname)
                                except ValueError:
                                    continue

                                if dt == day_dt:
                                    closest_file = fname
                                    exact_found = True
                                    break

                                diff = abs((dt - day_dt).total_seconds())
                                if diff < closest_diff:
                                    closest_file = fname
                                    closest_diff = diff

                            if not closest_file:
                                continue

                            dt = extract_time(closest_file)
                            if not exact_found:
                                start_dt = datetime.strptime(f"{year}-{month}-{day:02d} {TIME_START}", "%Y-%m-%d %H_%M")
                                end_dt = datetime.strptime(f"{year}-{month}-{day:02d} {TIME_END}", "%Y-%m-%d %H_%M")
                                if not (start_dt <= dt <= end_dt):
                                    continue

                            os.makedirs(local_dir, exist_ok=True)
                            time_part = dt.strftime("%H_%M")
                            local_filename = f"{day_key}-{time_part}-{option.lower()}.jpeg"
                            local_path = os.path.join(local_dir, local_filename)
                            remote_file = f"{month_path}/{closest_file}"

                            if os.path.exists(local_path):
                                continue

                            try:
                                sftp.get(remote_file, local_path)
                                print(f"[DL] {remote_file} -> {local_path}")
                                executor.submit(validate_and_quarantine, local_path)
                            except Exception as e:
                                print(f"[ERROR] Download failed: {remote_file} -> {e}")

executor.shutdown(wait=True)
sftp.close()
client.close()
print("[DONE]")
