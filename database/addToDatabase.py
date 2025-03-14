import pyautogui
import time
import random
import os
import random

# Enable PyAutoGUI's fail-safe: moving the mouse to a corner aborts the program.
pyautogui.FAILSAFE = True

game_count = 1

START_GAME_COORDS = (1350, 690)  # Loc to start the first game
GUESS_COORDS = (1800, 1000)  # Guess loc
FINALIZE_COORDS = (1800, 1050)  # Make guess loc
NEXT_ROUND_COORDS = (1000, 1000)  # Go next loc


def create_game_folder(game_number):
    folder = f"C:/Users/fishd/PycharmProjects/Geoguessr AI/database/temp_storage/images/Game_{game_number}"
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder


def take_and_save_screenshot(folder, game_number, round_number):
    screenshot = pyautogui.screenshot()
    filename = os.path.join(folder, f"game{game_number}_round{round_number}.png")
    screenshot.save(filename)
    print(f"Saved screenshot: {filename}")


def wait_random(min_seconds, max_seconds):
    delay = random.uniform(min_seconds, max_seconds)
    print(f"Waiting {delay:.2f} seconds...")
    time.sleep(delay)


def play_game(game_number):
    print(f"\n=== Starting Game #{game_number} ===")
    folder = create_game_folder(game_number)

    # Loop for 5 rounds
    for round_number in range(1, 6):
        print(f"\n-- Round {round_number} --")
        wait_random(3, 10)
        take_and_save_screenshot(folder, game_number, round_number)
        pyautogui.click(*GUESS_COORDS)
        print(f"Clicked guess at {GUESS_COORDS}")
        time.sleep(0.5)
        pyautogui.click(*FINALIZE_COORDS)
        print(f"Clicked finalize at {FINALIZE_COORDS}")
        time.sleep(random.uniform(4,6))
        pyautogui.click(*NEXT_ROUND_COORDS)
        print(f"Clicked next round at {NEXT_ROUND_COORDS}")
        time.sleep(random.uniform(1,3))

    wait_random(1, 5)
    pyautogui.click(*NEXT_ROUND_COORDS)
    print(f"Clicked restart at {NEXT_ROUND_COORDS}")


def main():
    global game_count
    try:
        time.sleep(5)
        pyautogui.click(*START_GAME_COORDS)
        print(f"Clicked start game at {START_GAME_COORDS}")
        time.sleep(2)

        while True:
            if random.uniform(0, 1) < 0.0035:
                to_sleep = random.uniform(2500, 4500)
                time.sleep(to_sleep)
                print(f"Sleeping for {to_sleep / 60:.2f} minutes")
            if game_count % 2 == 0:
                pyautogui.press('f5')
                print("Refreshing")
                time.sleep(6)
            play_game(game_count)
            game_count += 1
            print(f"Game #{game_count - 1} complete. Preparing for next game...\n")
            time.sleep(2)
    except KeyboardInterrupt:
        print("Automation stopped by user.")


if __name__ == "__main__":
    main()
