# ABOUTME: Backward compatibility shim
# ABOUTME: Re-exports main from cli module for legacy entry point support

from sapat.cli import main

if __name__ == "__main__":
    main()
