# fetcher.py
import requests
from scripts.protocol_utils import write_logfile_entry

def fetch_data(ip):
    try:
        res = requests.get(f"http://{ip}/api/system/info", timeout=5)
        return res.json()
    except Exception as e:
        write_logfile_entry(f"[{ip}] API Error: {e}")
        return None
