import sys,os,curses

import curses
from curses import textpad

screen = curses.initscr()
curses.noecho()
curses.cbreak()
screen.keypad(True)

# def getkey():
#     old_settings = termios.tcgetattr(sys.stdin)
#     tty.setcbreak(sys.stdin.fileno())
#     try:
#         while True:
#             b = os.read(sys.stdin.fileno(), 3).decode()
#             if len(b) == 3:
#                 k = ord(b[2])
#             else:
#                 k = ord(b)
#             key_mapping = {
#                 127: 'backspace',
#                 10: 'return',
#                 32: 'space',
#                 9: 'tab',
#                 27: 'esc',
#                 65: 'up',
#                 66: 'down',
#                 67: 'right',
#                 68: 'left'
#             }
#             return key_mapping.get(k, chr(k))
#     finally:
#         termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

if __name__ == "__main__":
    try:
        # print("Press a key")
        char = screen.getch()
        # while True:
        if char == ord('q'):
            print("q")
            # break
        elif char == curses.KEY_UP:
            print('up')
        elif char == curses.KEY_DOWN:
            print('down')
        elif char == curses.KEY_RIGHT:
            print('right')
        elif char == curses.KEY_LEFT:
            print('left')
        elif char == ord('s'):
            print('stop')

    finally:
        curses.nocbreak(); screen.keypad(0); curses.echo()
        curses.endwin()