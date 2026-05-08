# This file contains small functions to control code workflow
import sys

def pause():
    if input("Press enter to continue or 'q' to quit.\n") == "q":
        sys.exit()