"""
CLI Demo Application
Demonstrates the chat assistant with session memory and query understanding.
"""

import argparse
import os
import sys
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.markdown import Markdown

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _PROJECT_ROOT)
_DEFAULT_LOG_DIR = os.path.join(_PROJECT_ROOT, "conversation_logger", "cli")

from core import ChatAssistant
from utils.conversation_logger import ConversationLogger

console = Console()


def main():
    """Main CLI demo"""
    parser = argparse.ArgumentParser(description="Chat Assistant Demo (Gemini)")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Gemini model name (default: gemini-2.0-flash)"
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=1000,
        help="Token threshold for summarization"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to JSONL log file or directory for saving the conversation (default: conversation_logger/cli/)"
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Do not save conversation log (disable default logging to conversation_logger/cli/)"
    )
    parser.add_argument(
        "--load-log",
        type=str,
        default=None,
        help="Path to conversation log to load"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed query understanding analysis"
    )
    
    args = parser.parse_args()
    
    # Initialize assistant
    try:
        assistant = ChatAssistant(
            llm_provider="gemini",
            llm_model=args.model,
            token_threshold=args.threshold,
            use_tokenizer=True,
            keep_recent_after_summary=5,
            recent_messages_window=20,
        )
    except Exception as e:
        console.print(f"[red]Error initializing assistant: {e}[/red]")
        console.print("\n[bold]Make sure you have:[/bold]")
        console.print("  1. Set GEMINI_API_KEY (or GOOGLE_API_KEY) in .env")
        console.print("  2. Run: pip install -r requirements.txt")
        sys.exit(1)

    # Conversation log: default to conversation_logger/cli/ unless --no-log or --log-file overrides
    logger = None
    if not args.no_log:
        if args.log_file:
            log_path = os.path.abspath(args.log_file)
            if os.path.isdir(log_path):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_path = os.path.join(log_path, f"cli_conversation_{ts}.jsonl")
        else:
            os.makedirs(_DEFAULT_LOG_DIR, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = os.path.join(_DEFAULT_LOG_DIR, f"cli_{ts}.jsonl")
        logger = ConversationLogger(log_path)
        console.print(f"[dim]Conversation log: {log_path}[/dim]")
    
    # Load conversation log if provided
    if args.load_log:
        assistant.load_conversation_log(args.load_log)
    
    # Seed log file with loaded history so the log = loaded + new messages (same source for 20 recent)
    if logger and assistant.memory_manager.conversation_history:
        logger.seed_from_history(assistant.memory_manager.conversation_history)
    
    # Welcome message
    console.print(Panel(
        "[bold cyan]Chat Assistant Demo[/bold cyan]\n\n"
        "Pipeline: 20 recent messages → Summarize → Query Understanding → Answer\n\n"
        "Commands: 'exit'/'quit' to exit | 'summary' to trigger summarization | 'stats' for context stats",
        title="Welcome",
        border_style="cyan"
    ))
    console.print()
    
    # Main loop
    while True:
        try:
            user_input = Prompt.ask("[bold green]You[/bold green]")
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                console.print("[yellow]Goodbye![/yellow]")
                break
            
            if user_input.lower() == 'summary':
                if assistant.memory_manager.conversation_history:
                    console.print("[yellow]Manually triggering summarization...[/yellow]")
                    summary = assistant.memory_manager.summarize_conversation(assistant.llm_client)
                    assistant.display_summary(summary)
                else:
                    console.print("[red]No conversation history to summarize[/red]")
                continue
            
            if user_input.lower() == 'stats':
                context_size = assistant.memory_manager.get_context_size()
                threshold = assistant.memory_manager.token_threshold
                console.print(f"\n[bold]Context Statistics:[/bold]")
                console.print(f"  Current size: {context_size} tokens")
                console.print(f"  Threshold: {threshold} tokens")
                console.print(f"  Usage: {context_size/threshold*100:.1f}%")
                console.print(f"  Messages: {len(assistant.memory_manager.conversation_history)}")
                if assistant.memory_manager.current_summary:
                    console.print(f"  Has summary: Yes (ID: {assistant.memory_manager.current_summary.memory_id})")
                else:
                    console.print(f"  Has summary: No")
                console.print()
                continue
            
            if not user_input.strip():
                continue
            
            # Process message
            result = assistant.process_user_message(user_input)
            
            # Display query understanding if verbose
            if args.verbose:
                assistant.display_query_understanding(result["query_understanding"])
            
            # Log user message
            if logger:
                logger.log_message("user", user_input)

            # Display response
            console.print(Panel(
                Markdown(result["response"]),
                title="[bold blue]Assistant[/bold blue]",
                border_style="blue"
            ))

            # Log assistant response (with basic metadata)
            if logger:
                metadata = {
                    "query_understanding": (
                        result["query_understanding"].model_dump()
                        if hasattr(result["query_understanding"], "model_dump")
                        else result["query_understanding"]
                    ),
                    "summary_triggered": result["summary_triggered"],
                    "context_size": result["context_size"],
                }
                if result["summary"]:
                    metadata["summary_id"] = getattr(result["summary"], "memory_id", None)
                logger.log_message("assistant", result["response"], metadata=metadata)
            
            # Show summary if triggered
            if result["summary_triggered"] and result["summary"]:
                assistant.display_summary(result["summary"])
            
            # Show context size warning
            context_size = result["context_size"]
            threshold = assistant.memory_manager.token_threshold
            if context_size > threshold * 0.8:
                console.print(f"[yellow]⚠ Context size: {context_size}/{threshold} tokens ({context_size/threshold*100:.1f}%)[/yellow]\n")
            else:
                console.print(f"[dim]Context: {context_size}/{threshold} tokens[/dim]\n")
        
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Type 'exit' to quit.[/yellow]\n")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]\n")


if __name__ == "__main__":
    main()
