"""wev <command>: serve | export | evaluate | train"""
import sys


def main():
    commands = {"serve": "wev.serve", "export": "wev.export", "evaluate": "wev.evaluate",
                "train": "wev.train"}
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print(__doc__)
        sys.exit(2)
    import importlib
    cmd = sys.argv.pop(1)
    sys.argv[0] = f"wev {cmd}"
    importlib.import_module(commands[cmd]).main()


if __name__ == "__main__":
    main()
