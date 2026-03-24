from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import mss
import numpy as np
import pyautogui
import pytesseract

TZ = ZoneInfo("Asia/Jerusalem")
ASSETS = Path("./assets")
APP_NAME = "ארלוזורוב 97"

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.01

CONF_TITLE = 0.84
CONF_BUTTON = 0.84
CONF_DAY = 0.82
SCROLL_STEP = -650
MAX_SCROLLS = 8

# ריצות אמת:
# שבת 07:59:59.900 -> ראשון 08:00
# ראשון 08:44:59.900 -> שני 08:45
# שלישי 08:44:59.900 -> רביעי 08:45
# רביעי 08:44:59.900 -> חמישי 08:45
RUNS = {
    5: {"target_day": "sun", "target_time": "08:00", "open_h": 8, "open_m": 0},
    6: {"target_day": "mon", "target_time": "08:45", "open_h": 8, "open_m": 45},
    1: {"target_day": "wed", "target_time": "08:45", "open_h": 8, "open_m": 45},
    2: {"target_day": "thu", "target_time": "08:45", "open_h": 8, "open_m": 45},
}

DAY_IMAGE = {
    "sun": "day_sun.png",
    "mon": "day_mon.png",
    "wed": "day_wed.png",
    "thu": "day_thu.png",
}


def now_local() -> dt.datetime:
    return dt.datetime.now(TZ)


def log(msg: str) -> None:
    print(f"[{now_local().strftime('%H:%M:%S.%f')[:-3]}] {msg}", flush=True)


def asset(name: str) -> str:
    p = ASSETS / name
    if not p.exists():
        raise FileNotFoundError(f"Missing asset: {p}")
    return str(p)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Counter booking bot")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run test mode immediately (or after delay) instead of waiting for the real schedule.",
    )
    parser.add_argument(
        "--time",
        type=str,
        help='Class time to look for, e.g. "08:00" or "08:45". Required in --test mode.',
    )
    parser.add_argument(
        "--day",
        choices=["sun", "mon", "wed", "thu"],
        help="Day tab to choose in test mode. Optional but recommended.",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=5.0,
        help="In test mode, wait this many seconds before starting. Default: 5",
    )
    return parser.parse_args()


def validate_time_string(value: str) -> bool:
    try:
        dt.datetime.strptime(value, "%H:%M")
        return True
    except ValueError:
        return False


def infer_test_day(class_time: str) -> str:
    # אם לא הועבר day בטסט:
    # 08:00 -> ראשון
    # 08:45 -> שני (ברירת מחדל נוחה לטסט)
    if class_time == "08:00":
        return "sun"
    if class_time == "08:45":
        return "mon"
    return "sun"


def sleep_until(target: dt.datetime) -> None:
    while True:
        remaining = (target - now_local()).total_seconds()
        if remaining <= 0:
            return
        time.sleep(min(remaining / 2, 0.5))


def precise_wait_until(target_perf: float) -> None:
    while time.perf_counter() < target_perf:
        pass


def open_app() -> None:
    subprocess.run(["open", "-a", APP_NAME], check=False)
    time.sleep(1.5)
    subprocess.run(
        ["osascript", "-e", f'tell application "{APP_NAME}" to activate'],
        check=False,
    )
    time.sleep(0.8)


def screenshot_rgb() -> np.ndarray:
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        img = np.array(sct.grab(monitor))[:, :, :3]
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def locate_all(image_name: str, confidence: float):
    img = screenshot_rgb()
    template_bgr = cv2.imread(asset(image_name))
    if template_bgr is None:
        raise FileNotFoundError(f"Could not read image {asset(image_name)}")
    template = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2RGB)

    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    ys, xs = np.where(res >= confidence)

    h, w = template.shape[:2]
    seen = []
    out = []

    for x, y in zip(xs, ys):
        keep = True
        for sx, sy in seen:
            if abs(x - sx) < 30 and abs(y - sy) < 20:
                keep = False
                break
        if keep:
            seen.append((x, y))
            out.append((x, y, w, h))

    return out


def click_center(box) -> None:
    x, y, w, h = box
    cx = x + w // 2
    cy = y + h // 2
    pyautogui.moveTo(cx, cy, duration=0)
    pyautogui.click()


def click_image_once(image_name: str, confidence: float = 0.84) -> bool:
    boxes = locate_all(image_name, confidence)
    if not boxes:
        return False
    click_center(boxes[0])
    return True


def reset_scroll_to_top() -> None:
    for _ in range(6):
        pyautogui.scroll(1000)
        time.sleep(0.05)


def preprocess_for_ocr(img_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)[1]
    return gray


def region_text(img_rgb: np.ndarray) -> str:
    proc = preprocess_for_ocr(img_rgb)
    txt = pytesseract.image_to_string(proc, config="--psm 6")
    return txt.replace(" ", "").strip()


def box_for_target_class(target_time: str):
    """
    מוצא את כרטיס 'חדר כושר' עם השעה הנכונה:
    1) מאתר את כל המופעים של title_gym.png
    2) בודק באזור העליון של אותו כרטיס אם מופיעה השעה הרצויה
    """
    titles = locate_all("title_gym.png", CONF_TITLE)
    if not titles:
        return None

    full = screenshot_rgb()
    screen_h, screen_w = full.shape[:2]

    for x, y, w, h in titles:
        rx1 = max(0, x - 260)
        ry1 = max(0, y - 110)
        rx2 = min(screen_w, x + 220)
        ry2 = min(screen_h, y - 20)

        time_region = full[ry1:ry2, rx1:rx2]
        txt = region_text(time_region)

        normalized_text = txt.replace(":", "")
        normalized_target = target_time.replace(":", "")
        if normalized_target in normalized_text:
            card_x1 = max(0, x - 300)
            card_y1 = max(0, y - 150)
            card_x2 = min(screen_w, x + 350)
            card_y2 = min(screen_h, y + 180)
            return (card_x1, card_y1, card_x2 - card_x1, card_y2 - card_y1)

    return None


def find_class_with_scroll(target_time: str):
    reset_scroll_to_top()
    for i in range(MAX_SCROLLS):
        box = box_for_target_class(target_time)
        if box:
            log(f"Found target class {target_time} after {i} scroll(s)")
            return box
        pyautogui.scroll(SCROLL_STEP)
        time.sleep(0.25)
    return None


def find_register_button():
    boxes = locate_all("btn_register.png", CONF_BUTTON)
    if not boxes:
        return None
    return boxes[0]


def go_to_schedule() -> None:
    boxes = locate_all("tab_shiurim.png", 0.3)
    log(f"tab_shiurim matches: {boxes}")

    for _ in range(3):
        if click_image_once("tab_shiurim.png", confidence=0.65):
            time.sleep(0.8)
            return
        time.sleep(0.2)

    raise RuntimeError("Could not click שיעורים tab")


def choose_day(day_key: str) -> None:
    for _ in range(3):
        if click_image_once(DAY_IMAGE[day_key], confidence=CONF_DAY):
            time.sleep(0.6)
            return
        time.sleep(0.2)
    raise RuntimeError(f"Could not click day tab: {day_key}")


def next_target():
    now = now_local()

    for add_days in range(0, 8):
        d = now + dt.timedelta(days=add_days)
        wd = d.weekday()
        if wd not in RUNS:
            continue

        cfg = RUNS[wd]
        candidate = dt.datetime(
            d.year,
            d.month,
            d.day,
            cfg["open_h"],
            cfg["open_m"],
            0,
            tzinfo=TZ,
        )

        if candidate > now:
            return candidate, cfg

    raise RuntimeError("No next target found")


def build_test_target(args: argparse.Namespace):
    if not args.time:
        raise ValueError("--time is required when using --test")

    if not validate_time_string(args.time):
        raise ValueError('--time must be in HH:MM format, for example "08:00"')

    day_key = args.day if args.day else infer_test_day(args.time)
    target_dt = now_local() + dt.timedelta(seconds=max(args.delay_seconds, 0.0))

    cfg = {
        "target_day": day_key,
        "target_time": args.time,
        "open_h": target_dt.hour,
        "open_m": target_dt.minute,
    }
    return target_dt, cfg


def attempt_booking(target_dt: dt.datetime, cfg: dict, test_mode: bool = False) -> None:
    day_key = cfg["target_day"]
    target_time = cfg["target_time"]

    prepare_dt = target_dt - dt.timedelta(seconds=90)
    card_click_dt = target_dt - dt.timedelta(milliseconds=100)

    log(
        f"Run mode={'TEST' if test_mode else 'LIVE'} | "
        f"target_dt={target_dt.isoformat()} | day={day_key} | class={target_time}"
    )

    if prepare_dt > now_local():
        sleep_until(prepare_dt)

    open_app()
    go_to_schedule()
    choose_day(day_key)

    class_box = find_class_with_scroll(target_time)
    if not class_box:
        raise RuntimeError(f"Could not find חדר כושר {target_time}")

    x, y, w, h = class_box
    cx = x + w // 2
    cy = y + h // 2
    pyautogui.moveTo(cx, cy, duration=0)
    log(f"Cursor parked on target class {target_time}")

    delta = (card_click_dt - now_local()).total_seconds()
    if delta > 0.3:
        time.sleep(delta - 0.2)

    target_perf = time.perf_counter() + max(
        0.0, (card_click_dt - now_local()).total_seconds()
    )
    precise_wait_until(target_perf)

    pyautogui.click()
    log("Class card clicked")

    end_time = time.perf_counter() + 4.0
    clicked = False

    while time.perf_counter() < end_time:
        btn = find_register_button()
        if btn:
            click_center(btn)
            time.sleep(0.03)
            pyautogui.click()
            clicked = True
            log("Register clicked")
            break

    if not clicked:
        raise RuntimeError("Register button not found in time")

    log("Done")


def main():
    args = parse_args()

    if args.test:
        target_dt, cfg = build_test_target(args)
        attempt_booking(target_dt=target_dt, cfg=cfg, test_mode=True)
        return

    target_dt, cfg = next_target()
    attempt_booking(target_dt=target_dt, cfg=cfg, test_mode=False)


if __name__ == "__main__":
    main()
