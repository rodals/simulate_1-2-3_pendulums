#!/usr/bin/python3
import sys
sys.path.insert(0, './modules')
import menu

all_args = sys.argv[1:]
if __name__ == '__main__':
    if len(all_args) == 0:
        menu.complete_menu()
    elif len(all_args) == 1:
        menu.input_menu(all_args[0])
    elif len(all_args) > 1:
        for arg in all_args:
            menu.input_menu(arg)

