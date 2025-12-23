import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: cm <command> [args]")
        print("Commands: calculate, chart")
        sys.exit(1)

    command = sys.argv[1]
    sys.argv = sys.argv[1:]  # Shift args so subcommand sees itself as argv[0]

    if command == "calculate":
        from .calculate import main as calculate_main

        calculate_main()
    elif command == "chart":
        from .chart import main as chart_main

        chart_main()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
