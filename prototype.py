import random
import time

def detect_conditions(speed, location_status):
    """Decide mode based on speed and location."""
    if speed > 20 and location_status == "on road":
        return "Drive Mode"
    elif speed <= 20 and location_status == "off road":
        return "Park Mode"
    else:
        return "Idle"

# Prototype simulation
print("=== Prototype Vehicle Simulation ===\n")

for i in range(5):  # simulate 5 readings
    speed = random.randint(0, 100)  # random speed between 0–100 km/h
    location_status = random.choice(["on road", "off road"])  # random location
    mode = detect_conditions(speed, location_status)

    print(f"Speed: {speed} km/h -> Mode: {mode} (Location: {location_status})")
    time.sleep(1)
