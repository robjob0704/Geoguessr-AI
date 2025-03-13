import pyautogui
import time
import random
import os
import random

# Enable PyAutoGUI's fail-safe: moving the mouse to a corner aborts the program.
pyautogui.FAILSAFE = True

# Set up a game counter
game_count = 1

# Define coordinates for each step (you may adjust these based on your monitor/setup)
START_GAME_COORDS = (1350, 690)  # Click to start the first game.
GUESS_COORDS = (1800, 1000)  # Click to place your guess.
FINALIZE_COORDS = (1800, 1050)  # Click to finalize your guess.
NEXT_ROUND_COORDS = (1000, 1000)  # Click to go to the next round (and later to restart).


def create_game_folder(game_number):
    folder = f"C:/Users/fishd/PycharmProjects/Geoguessr AI/database/temp_storage/images/Game_{game_number}"
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder


def take_and_save_screenshot(folder, game_number, round_number):
    # Capture the entire screen; you could also specify a region if desired.
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

    # Loop for 5 rounds.
    for round_number in range(1, 6):
        print(f"\n-- Round {round_number} --")
        wait_random(3, 10)  # Wait 1 to 10 seconds.
        take_and_save_screenshot(folder, game_number, round_number)
        # Click to guess the location on the map.
        pyautogui.click(*GUESS_COORDS)
        print(f"Clicked guess at {GUESS_COORDS}")
        time.sleep(0.5)
        # Finalize the guess.
        pyautogui.click(*FINALIZE_COORDS)
        print(f"Clicked finalize at {FINALIZE_COORDS}")
        time.sleep(random.uniform(4,6))
        # Proceed to next round.
        pyautogui.click(*NEXT_ROUND_COORDS)
        print(f"Clicked next round at {NEXT_ROUND_COORDS}")
        time.sleep(random.uniform(1,3))

    # After the fifth round, wait 1 to 5 seconds, then click again at the next round coordinates to restart the game.
    wait_random(1, 5)
    pyautogui.click(*NEXT_ROUND_COORDS)
    print(f"Clicked restart at {NEXT_ROUND_COORDS}")


def main():
    global game_count
    try:
        # Start the game by clicking the start button.
        time.sleep(5)
        pyautogui.click(*START_GAME_COORDS)
        print(f"Clicked start game at {START_GAME_COORDS}")
        time.sleep(2)  # Allow a couple of seconds for the game to start.

        while True:
            if game_count % 2 == 0:
                pyautogui.press('f5')
                print("Refreshing")
                time.sleep(6)
            play_game(game_count)
            game_count += 1
            # Optionally, add a delay before starting the next game
            print(f"Game #{game_count - 1} complete. Preparing for next game...\n")
            time.sleep(2)
    except KeyboardInterrupt:
        print("Automation stopped by user.")


if __name__ == "__main__":
    main()
