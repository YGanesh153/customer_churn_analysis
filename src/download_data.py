from pathlib import Path
from urllib.request import urlopen
import hashlib
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
EXPECTED_SHA256 = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading the IBM Telco Customer Churn sample dataset...")
    try:
        with urlopen(URL, timeout=45) as response:
            content = response.read()
    except Exception as exc:
        raise SystemExit(
            f"Download failed: {exc}\n"
            "Download the CSV manually from the IBM repository and save it as:\n"
            f"{OUT}"
        )
    digest = hashlib.sha256(content).hexdigest()
    if digest != EXPECTED_SHA256:
        raise SystemExit(
            "Dataset checksum did not match the expected IBM sample file. "
            "The downloaded file was not saved. Please inspect the source."
        )
    OUT.write_bytes(content)
    print(f"Saved: {OUT}")
    print(f"SHA-256 verified: {digest}")

if __name__ == "__main__":
    main()
