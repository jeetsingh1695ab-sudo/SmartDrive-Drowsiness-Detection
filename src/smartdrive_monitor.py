# SmartDrive - Driver Monitoring

import time

DROWSY_TIME = 2.0


def detect_drowsiness(left_state, right_state, closed_start):
    both_closed = (
        left_state == "CLOSED"
        and right_state == "CLOSED"
    )

    if not both_closed:
        return "SAFE", None

    if closed_start is None:
        closed_start = time.time()

    duration = time.time() - closed_start

    if duration >= DROWSY_TIME:
        return "DROWSY", closed_start

    return "WARNING", closed_start


def get_closed_duration(closed_start):
    if closed_start is None:
        return 0.0

    return time.time() - closed_start


def reset_monitor():
    return None


if __name__ == "__main__":
    print("SmartDrive Driver Monitoring")
    print("Drowsiness threshold:", DROWSY_TIME, "seconds")
