import os
import re
import time
import csv
import shutil
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Define directories (update paths if needed)
COORDINATES_DIR = r"C:\Users\fishd\PycharmProjects\Geoguessr AI\database\temp_storage\coordinates"
IMAGES_BASE_DIR = r"C:\Users\fishd\PycharmProjects\Geoguessr AI\database\temp_storage\images"
COMPLETED_DIR = r"C:\Users\fishd\PycharmProjects\Geoguessr AI\database\temp_storage\completed"


def process_csv(csv_path):
    print(f"\nProcessing CSV file: {csv_path}")
    # Extract game number from filename (expected format: geoguessr-game-{n}-locations.csv)
    base_filename = os.path.basename(csv_path)
    match = re.search(r"geoguessr-game-(\d+)-locations\.csv", base_filename)
    if not match:
        print("CSV filename doesn't match expected format:", base_filename)
        return
    game_number = match.group(1)
    print(f"Extracted game number: {game_number}")

    # Read the CSV file and build a mapping: round number -> (lat, lng)
    mapping = {}
    try:
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    round_num = int(row['Round'])
                    lat = float(row['Lat'])
                    lng = float(row['Lng'])
                    mapping[round_num] = (lat, lng)
                except Exception as e:
                    print(f"Error parsing row {row}: {e}")
    except Exception as e:
        print("Error reading CSV file:", e)
        return
    print("Round mapping:", mapping)

    # Determine the images directory for this game
    images_dir = os.path.join(IMAGES_BASE_DIR, f"Game_{game_number}")
    if not os.path.isdir(images_dir):
        print(f"Images directory not found: {images_dir}")
        return

    # Process each PNG in the images folder
    for filename in os.listdir(images_dir):
        if filename.lower().endswith(".png"):
            # Assume the file name includes "round{n}" e.g., game1_round5.png
            round_match = re.search(r"round(\d+)", filename, re.IGNORECASE)
            if not round_match:
                print(f"Could not determine round for file: {filename}")
                continue
            round_num = int(round_match.group(1))
            if round_num not in mapping:
                print(f"Round {round_num} not found in CSV mapping for file: {filename}")
                continue

            lat, lng = mapping[round_num]
            # Create a new filename using the coordinates (round to 8 decimals)
            new_filename = f"usa_{lat:.8f}_{lng:.8f}.png"
            src_path = os.path.join(images_dir, filename)
            dst_path = os.path.join(COMPLETED_DIR, new_filename)
            try:
                shutil.copy2(src_path, dst_path)
                print(f"Copied '{src_path}' -> '{dst_path}'")
            except Exception as e:
                print(f"Error copying '{src_path}' to '{dst_path}': {e}")

    # After processing, perform cleanup
    cleanup_files(game_number, csv_path)


def cleanup_files(game_number, csv_path):
    # Delete the images folder for this game
    images_dir = os.path.join(IMAGES_BASE_DIR, f"Game_{game_number}")
    try:
        if os.path.exists(images_dir):
            shutil.rmtree(images_dir)
            print(f"Deleted images folder: {images_dir}")
        else:
            print(f"Images folder not found for cleanup: {images_dir}")
    except Exception as e:
        print(f"Error deleting images folder {images_dir}: {e}")

    # Delete the CSV file
    try:
        if os.path.exists(csv_path):
            os.remove(csv_path)
            print(f"Deleted CSV file: {csv_path}")
        else:
            print(f"CSV file not found for cleanup: {csv_path}")
    except Exception as e:
        print(f"Error deleting CSV file {csv_path}: {e}")


def reset_round_data():
    # In this script, the round data is local in the CSV file.
    # Once processing is complete, no persistent data is kept.
    print("Round data processing complete.\n")


class CSVHandler(FileSystemEventHandler):
    def __init__(self):
        super().__init__()
        self.processed_files = set()

    def process_event(self, event):
        if not event.is_directory and event.src_path.lower().endswith(".csv"):
            if event.src_path in self.processed_files:
                return
            self.processed_files.add(event.src_path)
            print(f"Detected new CSV file: {event.src_path}")
            time.sleep(15)  # Wait 15 seconds to ensure the file is fully written
            process_csv(event.src_path)
            reset_round_data()

    def on_created(self, event):
        print("on_created event:", event.src_path)
        self.process_event(event)

    def on_modified(self, event):
        print("on_modified event:", event.src_path)
        self.process_event(event)


if __name__ == "__main__":
    # Ensure the completed directory exists
    if not os.path.exists(COMPLETED_DIR):
        os.makedirs(COMPLETED_DIR)
    print(f"Watching '{COORDINATES_DIR}' for new CSV files...")
    observer = Observer()
    observer.schedule(CSVHandler(), path=COORDINATES_DIR, recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping observer...")
        observer.stop()
    observer.join()
