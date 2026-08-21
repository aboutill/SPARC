import sys
from importlib.metadata import entry_points


def _discover():
    eps = entry_points(group="console_scripts")
    return {
        ep.name: ep
        for ep in eps
        if ep.module.startswith("sparc") and ep.name != "sparc"
    }


def main():
    commands = _discover()

    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("sparc: available commands\n")
        for name in sorted(commands):
            print(f"  {name}")
        print("\nRun 'sparc <command> -h' for help on a specific command.")
        return 0

    cmd = sys.argv[1]
    if cmd not in commands:
        print(f"Unknown command: {cmd!r}\n")
        print("Available commands:", ", ".join(sorted(commands)))
        return 1

    # dispatch: make the subcommand see a clean argv, then call it
    func = commands[cmd].load()
    sys.argv = [cmd] + sys.argv[2:]
    return func()