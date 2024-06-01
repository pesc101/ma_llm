from rich import print


def print_rich(text):
    print("=" * 80)
    print(f"[bold][green]{text}[/bold][/green]")
    print("=" * 80)
